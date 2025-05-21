from __future__ import annotations
import os
import re
import logging
from typing import Iterable, Union, Dict, List
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger(__name__)
__all__ = ["load_model_and_tokenizer", "embed_dataset"]

# skip pure punctuation/symbol tokens
_SKIP_PUNCT_RE = re.compile(r"^[^A-Za-z0-9]+$")
# skip pandas duplicate columns: optional non-alnum → dot → digits
_SKIP_DUP_RE   = re.compile(r"^[^A-Za-z0-9]*\.[0-9]+$")

def load_model_and_tokenizer(
    model_name: str = "google/gemma-2-9b-it",
    *,
    device: str | int | None = None,
    use_data_parallel: bool = True,
):
    """Load and cache model + tokenizer, optionally using DataParallel."""
    LOGGER.info("Loading model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval()

    if device is not None:
        model = model.to(device)

    if use_data_parallel and torch.cuda.device_count() > 1:
        LOGGER.info("Using DataParallel with %d GPUs", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    return model, tokenizer

def _get_blocks(model):
    """Return the list of transformer blocks."""
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    raise AttributeError("Cannot locate transformer layers.")

def _register_block_hooks(model) -> tuple[
    List[torch.utils.hooks.RemovableHandle],
    Dict[int, torch.Tensor],
    Dict[int, torch.Tensor],
]:
    """Attach hooks to capture attention and MLP outputs."""
    attn_cache: Dict[int, torch.Tensor] = {}
    mlp_cache: Dict[int, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for idx, block in enumerate(_get_blocks(model)):
        def make_hook(cache: Dict[int, torch.Tensor], i: int):
            def hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                cache[i] = t.detach().cpu().squeeze(0)
            return hook

        handles.append(block.self_attn.register_forward_hook(make_hook(attn_cache, idx)))
        handles.append(block.mlp.register_forward_hook(make_hook(mlp_cache, idx)))

    return handles, attn_cache, mlp_cache

def _find_target_token(tokens: List[str]) -> tuple[int, str]:
    """
    Find last valid token:
      - skip any 'end_of_turn'
      - skip pure punctuation/symbol tokens
      - skip pandas-style '.1', '_.2', etc.
    """
    for i in range(len(tokens) - 1, -1, -1):
        tok = tokens[i]
        if "end_of_turn" in tok:
            continue
        if _SKIP_PUNCT_RE.match(tok):
            continue
        if _SKIP_DUP_RE.match(tok):
            continue
        return i, tok
    return len(tokens) - 1, tokens[-1]

def embed_dataset(
    data: Union[pd.DataFrame, str, Iterable[str]],
    *,
    input_col: str | None = None,
    model_name: str = "google/gemma-2-9b-it",
    output_dir: str = "embeddings",
    device: str | None = "cuda:0",
    layers: List[int] | None = None,
    parts: List[str] | None = None,
    eos_token: str | None = None,
):
    """
    Generalized embedding extraction for HuggingFace models.
    Extracts last-token vectors for specified layers/streams.
    Saves CSVs to embeddings/{part}/{row_id}_L{layer:02d}.csv

    Parameters:
        data: DataFrame, path to CSV, or iterable of strings.
        input_col: Column name containing pre-formatted model inputs.
        model_name: HuggingFace model identifier.
        output_dir: Directory to save embeddings.
        device: Device to run model on.
        layers: List of layer indices to extract embeddings from.
        parts: Which parts to extract ('rs', 'attn', 'mlp').
        eos_token: Optional end-of-sequence token to append.
    """
    if isinstance(data, (str, os.PathLike)):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame({"__input__": list(data)})
        input_col = "__input__"

    if input_col is None:
        raise ValueError("input_col must be provided when data is a DataFrame")

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    n_layers = len(_get_blocks(model))

    layers = layers if layers is not None else list(range(n_layers))
    parts  = parts  if parts  is not None else ["rs", "attn", "mlp"]

    handles, attn_cache, mlp_cache = _register_block_hooks(model)
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
            prompt = row[input_col]
            if eos_token:
                prompt += eos_token

            enc = tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}

            attn_cache.clear()
            mlp_cache.clear()

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)

            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
            tgt_idx, tgt_tok = _find_target_token(tokens)
            row_id = row.get("id") or f"row_{idx}"

            for layer in layers:
                if not (0 <= layer < n_layers):
                    raise ValueError(f"Layer index {layer} out of bounds (0-{n_layers-1})")

                vectors = {
                    "rs":   out.hidden_states[layer + 1][0, tgt_idx].cpu().numpy(),
                    "attn": attn_cache[layer][tgt_idx].numpy(),
                    "mlp":  mlp_cache[layer][tgt_idx].numpy(),
                }

                for part in parts:
                    if part not in vectors:
                        raise ValueError(f"Invalid part '{part}'.")
                    vec = vectors[part]
                    part_dir = os.path.join(output_dir, part)
                    os.makedirs(part_dir, exist_ok=True)
                    path = os.path.join(part_dir, f"{row_id}_L{layer:02d}.csv")
                    pd.DataFrame(vec.reshape(-1, 1), columns=[tgt_tok]).to_csv(path, index=False)
    finally:
        for h in handles:
            h.remove()



def embed(
    inputs: Union[str, List[str]],
    *,
    model_name: str = "google/gemma-2-9b-it",
    device: str | None = "cuda:0",
    layers: List[int] | None = None,
    parts: List[str] | None = None,
    eos_token: str | None = None,
) -> Dict[str, Dict[int, Dict[str, np.ndarray]]]:
    """
    Embed input string(s) and return last-token embeddings directly.

    Parameters:
        inputs: Single string or list of strings to embed.
        model_name: HuggingFace model identifier.
        device: Device to run model on.
        layers: List of layer indices to extract embeddings from.
        parts: Which parts to extract ('rs', 'attn', 'mlp').
        eos_token: Optional end-of-sequence token to append.

    Returns:
        Nested dictionary structure:
        {
            input_str: {
                layer_idx: {
                    part: embedding_vector (np.ndarray)
                }
            }
        }
    """
    import numpy as np

    if isinstance(inputs, str):
        inputs = [inputs]

    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    n_layers = len(_get_blocks(model))

    layers = layers if layers is not None else list(range(n_layers))
    parts  = parts  if parts  is not None else ["rs", "attn", "mlp"]

    handles, attn_cache, mlp_cache = _register_block_hooks(model)
    embeddings = {}

    try:
        for input_str in tqdm(inputs, desc="Embedding"):
            prompt = input_str + (eos_token if eos_token else "")

            enc = tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}

            attn_cache.clear()
            mlp_cache.clear()

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)

            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())
            tgt_idx, tgt_tok = _find_target_token(tokens)

            embeddings[input_str] = {}
            for layer in layers:
                if not (0 <= layer < n_layers):
                    raise ValueError(f"Layer index {layer} out of bounds (0-{n_layers-1})")

                vectors = {
                    "rs":   out.hidden_states[layer + 1][0, tgt_idx].cpu().numpy(),
                    "attn": attn_cache[layer][tgt_idx].numpy(),
                    "mlp":  mlp_cache[layer][tgt_idx].numpy(),
                }

                embeddings[input_str][layer] = {part: vectors[part] for part in parts}

    finally:
        for h in handles:
            h.remove()

    return embeddings