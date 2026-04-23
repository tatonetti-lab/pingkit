"""
Run inference on a novel question using a trained probe.

1. Takes a hardcoded user question
2. Uses Gemma 2 9B to generate an answer
3. Appends the answer to the prompt (ensures period at end)
4. Extracts embedding for the final period token
5. Runs the trained probe to get prediction

Example
-------
$ python probe_infer.py \
      --artifact_dir ./my_model \
      --question "Who received the IEEE Frank Rosenblatt Award in 2010?"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from contextlib import contextmanager, redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as tf_logging

from pingkit.embedding import embed
from pingkit.model import load_artifacts, _evaluate

# ---------------------- suppress noisy output ------------------------- #
tf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*weight.*")
warnings.filterwarnings("ignore", message=".*torch.classes.*")


@contextmanager
def suppress_stdout_stderr():
    """Context manager to swallow stdout/stderr (catches C-level loading spam)."""
    import sys
    with open(os.devnull, "w") as devnull, \
         redirect_stderr(devnull):
        yield


# ----------------------------- config --------------------------------- #
MODEL_NAME = "google/gemma-2-9b-it"
LAYERS = [39]
PARTS = ["rs"]
POOLING = "last"

# Gemma 2 prompt template
PROMPT_TEMPLATE = """<start_of_turn>user
Answer the following question in a single phrase:
{question}<end_of_turn>
<start_of_turn>model
"""


# --------------------------- cli helpers ------------------------------ #
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run probe inference on novel text.")
    p.add_argument(
        "--artifact_dir",
        type=Path,
        required=True,
        help="Directory containing trained probe artifacts.",
    )
    p.add_argument(
        "--question",
        type=str,
        default="Who received the IEEE Frank Rosenblatt Award in 2010?",
        help="Question to ask the model.",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=LAYERS,
        help="Layer(s) to extract embeddings from.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate for the answer.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device for model inference.",
    )
    return p.parse_args(argv)


# -------------------------- utility funcs ----------------------------- #
def generate_answer(
    prompt: str,
    model_name: str = MODEL_NAME,
    max_new_tokens: int = 50,
    device: str = "auto",
) -> str:
    """Generate an answer using the LLM."""
    print(f"Loading {model_name} for generation...")
    
    with suppress_stdout_stderr():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return generated.strip()




def concat_features_from_embed(emb_out: Dict, layers: List[int]) -> np.ndarray:
    """
    Extract and concatenate feature vectors from embed() output.
    Returns 1D np.ndarray.
    """
    vectors: List[np.ndarray] = []
    prompt_key = next(iter(emb_out.keys()))
    by_layer = emb_out[prompt_key]
    
    for layer in sorted(layers):
        layer_dict = by_layer[layer]
        for part in PARTS:
            pool_dict = layer_dict[part]
            # pool_dict keys may be token strings; take the first value
            vec = next(iter(pool_dict.values()))
            vectors.append(vec.reshape(-1))
    
    return np.concatenate(vectors, axis=0)


def build_dataframe(vec: np.ndarray, meta: Dict) -> pd.DataFrame:
    """
    Build DataFrame with feature columns matching the trained model.
    """
    n = vec.size
    cols = None
    for key in ("columns", "feature_columns", "feature_names"):
        if isinstance(meta.get(key), list) and len(meta[key]) == n:
            cols = meta[key]
            break
    
    if cols is None:
        cols = [f"f{i:04d}" for i in range(n)]
    
    df = pd.DataFrame([vec], columns=cols)
    df.index = pd.Index(["query_0"], name="id")
    return df


# ------------------------------- main --------------------------------- #
def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    
    # 1) Build the prompt
    prompt = PROMPT_TEMPLATE.format(question=args.question)
    print("=" * 60)
    print("QUESTION:")
    print(args.question)
    print("=" * 60)
    
    # 2) Generate answer with Gemma 2 9B
    print("\nGenerating answer...")
    answer = generate_answer(
        prompt,
        model_name=MODEL_NAME,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    print(f"\nGenerated answer: {answer}")
    
    
    # 4) Build full text for embedding (prompt + answer)
    full_text = prompt + answer
    print("\n" + "-" * 60)
    print("FULL TEXT FOR EMBEDDING:")
    print(full_text)
    print("-" * 60)
    
    # 5) Get embedding for the last token
    print(f"\nExtracting embeddings from layer(s) {args.layers}...")
    with suppress_stdout_stderr():
        emb = embed(
            full_text,
            model_name=MODEL_NAME,
            layers=args.layers,
            parts=PARTS,
            pooling=POOLING,
            device=args.device,
            filter_non_text=True,
        )
    
    # 6) Concatenate features
    vec = concat_features_from_embed(emb, args.layers)
    print(f"Feature vector shape: {vec.shape}")
    
    # 7) Load trained probe
    print(f"\nLoading probe from {args.artifact_dir}...")
    model, meta = load_artifacts(str(args.artifact_dir))
    
    # 8) Build DataFrame and run inference
    X = build_dataframe(vec, meta)
    X_np = X.values.astype(np.float32)
    dummy_y = np.zeros((X_np.shape[0],), dtype=np.int64)
    
    probs, _, _ = _evaluate(
        model,
        X_np,
        dummy_y,
        model_type="mlp",
        metric_fn=lambda y, p: 0.0,
        ce_loss=None,
        device=next(model.parameters()).device,
    )
    
    # 9) Load label encoder if available
    label_encoder_path = args.artifact_dir / "label_encoder.json"
    inv_labels = None
    if label_encoder_path.exists():
        with open(label_encoder_path, "r") as f:
            label_map = json.load(f)
        # label_map is {index: label_name}
        inv_labels = {int(k): v for k, v in label_map.items()}
    
    # 10) Print results
    print("\n" + "=" * 60)
    print("PROBE PREDICTION:")
    print("=" * 60)
    
    probs_row = probs[0]
    pred_idx = int(np.argmax(probs_row))
    
    for i, p in enumerate(probs_row):
        if inv_labels and i in inv_labels:
            print(f"  {inv_labels[i]}: {float(p):.4f}")
        else:
            print(f"  class_{i}: {float(p):.4f}")
    
    print()
    if inv_labels and pred_idx in inv_labels:
        print(f"Predicted: {inv_labels[pred_idx]} (prob={probs_row[pred_idx]:.4f})")
    else:
        print(f"Predicted: class_{pred_idx} (prob={probs_row[pred_idx]:.4f})")


if __name__ == "__main__":
    main()