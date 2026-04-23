#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def coerce_binary(series):
    # Robustly coerce to {0,1}
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)

def main():
    ap = argparse.ArgumentParser(description="Split eval results into refused, correct, incorrect CSVs and analyze base model performance.")
    ap.add_argument("--in_csv", required=True, help="Path to the eval results CSV (with is_refusal, is_correct columns).")
    ap.add_argument("--base_csv", required=True, help="Path to the base model results CSV (gemma_medmcqa_top_token.csv).")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: alongside input file).")
    ap.add_argument("--prefix", default=None, help="Optional filename prefix (default: input stem).")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    base_path = Path(args.base_csv)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix if args.prefix else in_path.stem

    # Load eval results
    df = pd.read_csv(in_path)
    
    # Ensure required columns exist
    for col in ["is_refusal", "is_correct"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {in_path}")

    # Coerce to 0/1 for robust filtering
    df["is_refusal"] = coerce_binary(df["is_refusal"])
    df["is_correct"] = coerce_binary(df["is_correct"])

    # Create splits
    refused = df[(df["is_refusal"] == 1) & (df["is_correct"] == 0)]
    correct = df[(df["is_refusal"] == 0) & (df["is_correct"] == 1)]
    incorrect = df[(df["is_refusal"] == 0) & (df["is_correct"] == 0)]

    # Save splits
    f_refused = out_dir / f"{prefix}_refused.csv"
    f_correct = out_dir / f"{prefix}_correct.csv"
    f_incorrect = out_dir / f"{prefix}_incorrect.csv"
    
    refused.to_csv(f_refused, index=False)
    correct.to_csv(f_correct, index=False)
    incorrect.to_csv(f_incorrect, index=False)

    print(f"Input: {in_path}")
    print(f"Refused: {len(refused):6d} -> {f_refused}")
    print(f"Correct: {len(correct):6d} -> {f_correct}")
    print(f"Incorrect: {len(incorrect):6d} -> {f_incorrect}")
    print()

    # Load base model results
    try:
        base_df = pd.read_csv(base_path)
        print(f"Base model results loaded from: {base_path}")
        
        # Ensure required columns exist in base model data
        required_base_cols = ["id", "correct", "top_token"]
        for col in required_base_cols:
            if col not in base_df.columns:
                raise ValueError(f"Missing required column '{col}' in base model CSV")
        
        # Calculate base model accuracy (correct == top_token)
        base_df["base_correct"] = (base_df["correct"] == base_df["top_token"]).astype(int)
        
        # Analyze base model performance within each split
        splits_data = {
            "refused": refused,
            "correct": correct, 
            "incorrect": incorrect
        }
        
        results = {}
        print("Base Model Performance Analysis:")
        print("=" * 50)
        
        for split_name, split_df in splits_data.items():
            if len(split_df) == 0:
                print(f"{split_name.capitalize():12s}: No samples")
                results[split_name] = {"total": 0, "base_correct": 0, "accuracy": 0.0}
                continue
                
            # Merge with base model results on 'id' column
            merged = split_df.merge(base_df[["id", "base_correct"]], on="id", how="inner")
            
            if len(merged) == 0:
                print(f"{split_name.capitalize():12s}: No matching IDs found in base model results")
                results[split_name] = {"total": 0, "base_correct": 0, "accuracy": 0.0}
                continue
            
            total_samples = len(merged)
            base_correct_count = merged["base_correct"].sum()
            accuracy = base_correct_count / total_samples if total_samples > 0 else 0.0
            
            results[split_name] = {
                "total": total_samples,
                "base_correct": base_correct_count,
                "accuracy": accuracy
            }
            
            print(f"{split_name.capitalize():12s}: {base_correct_count:4d}/{total_samples:4d} = {accuracy:.3f}")
        
        print()
        
        # Create stacked bar chart
        create_stacked_bar_chart(results, out_dir, prefix)
        
    except FileNotFoundError:
        print(f"Error: Base model CSV not found at {base_path}")
    except Exception as e:
        print(f"Error analyzing base model performance: {e}")

def create_stacked_bar_chart(results, out_dir, prefix):
    """Create a stacked bar chart showing correct/incorrect counts for each split."""
    
    # Prepare data for plotting
    splits = []
    correct_counts = []
    incorrect_counts = []
    
    for split_name, data in results.items():
        if data["total"] > 0:  # Only include splits with data
            splits.append(split_name.capitalize())
            correct_counts.append(data["base_correct"])
            incorrect_counts.append(data["total"] - data["base_correct"])
    
    if not splits:
        print("No data available for plotting.")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Colors as specified
    correct_color = "#37CC53"  # Green for correct
    incorrect_color = "#B45BF4"  # Purple for incorrect
    
    # Create stacked bars
    bar_width = 0.4
    x_pos = np.arange(len(splits))
    
    # Bottom bars (incorrect)
    bars1 = ax.bar(x_pos, incorrect_counts, bar_width, 
                   label='Incorrect', color=incorrect_color)
    
    # Top bars (correct)
    bars2 = ax.bar(x_pos, correct_counts, bar_width, 
                   bottom=incorrect_counts, label='Correct', color=correct_color)
    
    # Customize the plot
    #ax.set_xlabel('DPO-Trained Model Results', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('DPO-Trained Model Results', fontsize=14)#, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(splits)
    ax.legend(title="Base Model")
    
    # Add value labels on bars
    for i, (incorrect, correct) in enumerate(zip(incorrect_counts, correct_counts)):
        # Label for incorrect section (bottom)
        if incorrect > 0:
            ax.text(i, incorrect/2, str(incorrect), ha='center', va='center', 
                   color='white', fontweight='bold')
        
        # Label for correct section (top)
        if correct > 0:
            ax.text(i, incorrect + correct/2, str(correct), ha='center', va='center', 
                   color='white', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    chart_path = out_dir / f"{prefix}_base_model_performance.pdf"
    plt.savefig(chart_path, bbox_inches='tight')
    print(f"Chart saved to: {chart_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()

# label,color,generative_accuracy
# Gemma-2-2b-it,#53CAF9,0.548
# Gemma-2-9b-it,#37CC53,0.692
# Llama-3.1-8B,#ff5c95,0.407
# Llama-3.1-8B-Instruct,#B45BF4,0.610
# Llama-3.3-70B-Instruct,#4C78F2,0.788