#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Filter base model results to only incorrect answers.")
    ap.add_argument(
        "--in_csv",
        default="gemma_medmcqa_top_token.csv",
        help="Path to the base model CSV (default: gemma_medmcqa_top_token.csv)"
    )
    ap.add_argument(
        "--out_csv",
        default=None,
        help="Output CSV path (default: {input_stem}_incorrect_only.csv)"
    )
    args = ap.parse_args()

    in_path = Path(args.in_csv)

    # Default output name
    if args.out_csv:
        out_path = Path(args.out_csv)
    else:
        out_path = in_path.parent / f"{in_path.stem}_incorrect_only.csv"

    # Load the data
    df = pd.read_csv(in_path)

    # Check required columns
    required_cols = ["correct", "top_token"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {in_path}")

    # Filter to only incorrect answers
    incorrect_df = df[df["correct"] != df["top_token"]]

    # Save the filtered data
    incorrect_df.to_csv(out_path, index=False)

    # Print summary
    print(f"Input file: {in_path}")
    print(f"Total samples: {len(df):,}")
    print(f"Incorrect samples: {len(incorrect_df):,}")
    print(f"Error rate: {len(incorrect_df)/len(df):.3f}")
    print(f"Filtered CSV saved to: {out_path}")

if __name__ == "__main__":
    main()