#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

LETTER_FROM_IDX = {0: "A", 1: "B", 2: "C", 3: "D"}

def main():
    ap = argparse.ArgumentParser(description="Compute MLP accuracy on the subset where Gemma was correct.")
    ap.add_argument("--preds", default="artifacts/L36/predictions.csv",
                    help="Path to probe predictions.csv (must include id,true_label,pred_label).")
    ap.add_argument("--gemma", default="gemma_medmcqa_top_token.csv",
                    help="Path to Gemma generative results (id,correct,top_token).")
    ap.add_argument("--out", default=None,
                    help="Optional path to save the subset CSV (defaults to <preds_stem>_gemmaCorrectSubset.csv).")
    args = ap.parse_args()

    preds_path = Path(args.preds)
    gemma_path = Path(args.gemma)
    out_path = Path(args.out) if args.out else preds_path.with_name(preds_path.stem + "_gemmaCorrectSubset.csv")

    # Load; sep=None lets pandas sniff comma/tsv automatically.
    preds = pd.read_csv(preds_path)
    gemma = pd.read_csv(gemma_path, sep=None, engine="python")

    # Basic sanity checks
    for col in ["id", "true_label", "pred_label"]:
        if col not in preds.columns:
            raise ValueError(f"Missing '{col}' in {preds_path}")
    for col in ["id", "correct", "top_token"]:
        if col not in gemma.columns:
            raise ValueError(f"Missing '{col}' in {gemma_path}")

    # Normalize letter columns on gemma side
    norm = lambda s: s.astype(str).str.strip().str.upper()
    gemma["correct_norm"]   = norm(gemma["correct"])
    gemma["top_token_norm"] = norm(gemma["top_token"])

    # IDs where Gemma was correct
    gemma_correct_ids = set(gemma.loc[gemma["correct_norm"] == gemma["top_token_norm"], "id"])

    # Restrict predictions to overlapping IDs
    subset = preds[preds["id"].isin(gemma_correct_ids)].copy()

    if subset.empty:
        print("No overlapping IDs where Gemma was correct. Nothing to evaluate.")
        subset.to_csv(out_path, index=False)
        print(f"Saved empty subset to: {out_path}")
        return

    # Compute accuracy of the MLP on this subset
    # (true_label / pred_label are 0..3 as per your pipeline)
    subset["is_correct_subset"] = (subset["pred_label"].astype(int) == subset["true_label"].astype(int)).astype(int)

    # Add human-readable letters for convenience
    subset["true_letter"] = subset["true_label"].map(LETTER_FROM_IDX)
    subset["pred_letter"] = subset["pred_label"].map(LETTER_FROM_IDX)

    acc = subset["is_correct_subset"].mean()

    # Save subset
    subset.to_csv(out_path, index=False)

    # Print a concise report
    print(f"Probe predictions: {len(preds)} rows")
    print(f"Gemma results:     {len(gemma)} rows")
    print(f"Gemma-correct IDs: {len(gemma_correct_ids)}")
    print(f"Overlap subset:    {len(subset)} rows")
    print(f"MLP accuracy on Gemma-correct subset: {acc:.4f}")
    print(f"Saved subset to: {out_path}")

if __name__ == "__main__":
    main()
