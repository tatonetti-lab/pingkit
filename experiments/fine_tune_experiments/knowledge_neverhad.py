#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

LETTER_FROM_IDX = {0: "A", 1: "B", 2: "C", 3: "D"}
IDX_FROM_LETTER = {"A": 0, "B": 1, "C": 2, "D": 3}


def main():
    ap = argparse.ArgumentParser(
        description="Compute MLP accuracy on the subset where Gemma was INCORRECT."
    )
    ap.add_argument(
        "--preds",
        default="artifacts/L36/predictions.csv",
        help="Path to probe predictions.csv (must include id,true_label,pred_label).",
    )
    ap.add_argument(
        "--gemma",
        default="gemma_medmcqa_top_token.csv",
        help="Path to Gemma generative results (id,correct,top_token).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional path to save the subset CSV (defaults to <preds_stem>_gemmaIncorrectSubset.csv).",
    )
    args = ap.parse_args()

    preds_path = Path(args.preds)
    gemma_path = Path(args.gemma)
    out_path = (
        Path(args.out)
        if args.out
        else preds_path.with_name(preds_path.stem + "_gemmaIncorrectSubset.csv")
    )

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
    gemma["correct_norm"] = norm(gemma["correct"])
    gemma["top_token_norm"] = norm(gemma["top_token"])

    # IDs where Gemma was INCORRECT
    gemma_incorrect_ids = set(
        gemma.loc[gemma["correct_norm"] == gemma["top_token_norm"], "id"]
    )

    # Restrict predictions to overlapping IDs
    subset = preds[preds["id"].isin(gemma_incorrect_ids)].copy()

    if subset.empty:
        print("No overlapping IDs where Gemma was incorrect. Nothing to evaluate.")
        subset.to_csv(out_path, index=False)
        print(f"Saved empty subset to: {out_path}")
        return

    # Merge in Gemma's original (wrong) prediction for agreement analysis
    gemma_wrong = gemma.loc[
        gemma["id"].isin(gemma_incorrect_ids), ["id", "top_token_norm"]
    ].rename(columns={"top_token_norm": "gemma_pred_letter"})
    subset = subset.merge(gemma_wrong, on="id", how="left")

    # Map Gemma's letter prediction to index for comparison with pred_label
    subset["gemma_pred_idx"] = subset["gemma_pred_letter"].map(IDX_FROM_LETTER)

    # Add human-readable letters for convenience
    subset["true_letter"] = subset["true_label"].map(LETTER_FROM_IDX)
    subset["pred_letter"] = subset["pred_label"].map(LETTER_FROM_IDX)

    # Metric 1: Probe accuracy against ground truth
    subset["probe_correct"] = (
        subset["pred_label"].astype(int) == subset["true_label"].astype(int)
    ).astype(int)
    acc_vs_truth = subset["probe_correct"].mean()

    # Metric 2: Probe agreement with Gemma's original wrong answer
    subset["agrees_with_gemma"] = (
        subset["pred_label"].astype(int) == subset["gemma_pred_idx"].astype(int)
    ).astype(int)
    agreement_with_gemma = subset["agrees_with_gemma"].mean()

    # Save subset
    subset.to_csv(out_path, index=False)

    # Print report
    print(f"Probe predictions:      {len(preds)} rows")
    print(f"Gemma results:          {len(gemma)} rows")
    print(f"Gemma-incorrect IDs:    {len(gemma_incorrect_ids)}")
    print(f"Overlap subset:         {len(subset)} rows")
    print(f"")
    print(f"--- Results on Gemma-INCORRECT subset ---")
    print(f"Probe accuracy (vs ground truth):       {acc_vs_truth:.4f}")
    print(f"Probe agreement (with Gemma's wrong):   {agreement_with_gemma:.4f}")
    print(f"Chance (4-way):                         0.2500")
    print(f"")
    print(f"Saved subset to: {out_path}")


if __name__ == "__main__":
    main()