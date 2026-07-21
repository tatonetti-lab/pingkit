# scripts/run_mcq_probe.py

from __future__ import annotations
import os
import math
from typing import List, Dict
from transformers import AutoTokenizer

from pingkit.embedding import _SKIP_PUNCT_RE, _SKIP_DUP_RE


from pingkit.generate import ProbeSpec, generate_with_probes

# Hardcoded multiple choice question
QUESTION = """<start_of_turn>user
What is the role of non-governmental organisations (NGOs) in the global defence trade?

A. Non-governmental organisations are the primary actors in modern arms control measures.
B. Non-governmental agencies are unable to access the resources and information needed to develop effective policy or action on arms control.
C. Non-governmental groups have played a significant and effective role in developing arms control measures in recent years.
D. There is a serious lack of involvement by non-governmental actors in controlling the global defence trade.

What is the letter of the correct answer?<end_of_turn>
<start_of_turn>model
Answer: """


# Update these to match your environment
PROBE_ARTIFACTS = "MMLU_Gemma-2-9b-it_train/artifacts/MLP/L41.pt"
MODEL_NAME = "google/gemma-2-9b-it"

# Class labels: index -> option letter
CLASS_LABELS = ["A", "B", "C", "D"]


def run_mcq_probe(question: str, *, k_classes: int, probe_path: str, model_name: str) -> Dict[str, float]:
    # Build one ProbeSpec per class index (reuse the same trained probe)
    probes = [
        ProbeSpec(
            name=f"class_{i}",
            model_path=probe_path,
            class_index=i,
            pooling="last",
        )
        for i in range(k_classes)
    ]

    out = generate_with_probes(
        question,
        probes=probes,
        evaluate_on="first",                # score on the user input only
        model_name=model_name,
        generation_kwargs={"max_new_tokens": 0},  # no generation needed
        return_details=True,
        filter_non_text=False,
        eos_token=None,
    )

    # Collect per-class probabilities from the "first" evaluation
    probs = {}
    for i in range(k_classes):
        key = f"class_{i}"
        p = out["probes"].get(key, {}).get("first", math.nan)
        probs[CLASS_LABELS[i]] = float(p)

    return probs


def main():
    probs = run_mcq_probe(
        QUESTION,
        k_classes=len(CLASS_LABELS),
        probe_path=PROBE_ARTIFACTS,
        model_name=MODEL_NAME,
    )

    # Pretty print
    print("Question:")
    print(QUESTION.strip(), end="\n\n")

    print("Per-class probabilities (probe on user input / 'first'):")
    for label in CLASS_LABELS:
        print(f"  {label}: {probs.get(label, float('nan')):.4f}")

    # Predicted answer = argmax
    best_label = max(CLASS_LABELS, key=lambda l: probs.get(l, float('-inf')))
    print(f"\nPredicted answer: {best_label}")

    # Show which token the probe used in 'first' mode (last valid user token)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tok(QUESTION, return_tensors="pt")
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())

    valid_idxs = [i for i, t in enumerate(tokens)
                if not _SKIP_PUNCT_RE.match(t)
                and not _SKIP_DUP_RE.match(t)]
    if not valid_idxs and tokens:
        valid_idxs = list(range(len(tokens)))

    if valid_idxs:
        last_idx = valid_idxs[-1]
        last_tok = tokens[last_idx]
        print(f"\nProbe token (first mode): idx={last_idx}, token={last_tok!r}")
    else:
        print("\nProbe token (first mode): <no valid tokens>")


    # Optional: if you also want to see how the probe behaves while generating an answer:
    stream_out = generate_with_probes(
        QUESTION,
        probes=[
            ProbeSpec(name=f"class_{i}", model_path=PROBE_ARTIFACTS, class_index=i)
            for i in range(len(CLASS_LABELS))
        ],
        evaluate_on="stream",              # evaluate once per generated token
        model_name=MODEL_NAME,
        generation_kwargs={"max_new_tokens": 64, "do_sample": False},
        return_details=True,
    )
    print("\nGenerated text:")
    print(stream_out["text"])
    print("\nStream probabilities (last value per class shown):")
    for i, label in enumerate(CLASS_LABELS):
        seq = stream_out["probes"][f"class_{i}"].get("stream", [])
        print(f"  {label}: {seq[-1]:.4f}" if seq else f"  {label}: N/A")


if __name__ == "__main__":
    main()
