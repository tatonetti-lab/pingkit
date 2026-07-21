#!/usr/bin/env python
"""
===================

Evaluates model performance across different training set sizes while keeping
the test set fixed. For each training size, performs multiple runs with different
random samples to compute mean/min/max metrics.

Example:
--------
$ python train_eval.py \
    --layer 41 \
    --train_dir data/train \
    --test_dir data/test \
    --train_response_df train_labels.csv \
    --test_response_df test_labels.csv \
    --n_runs 5 \
    --output results/train_size_results.json
"""

import pathlib
import os
import json
import argparse
from datetime import datetime
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pingkit.extraction import extract_token_vectors
from pingkit.model import (
    fit, load_npz_features, _evaluate, save_artifacts
)


# ── Argument Parser ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Evaluate model performance across different training sizes"
)

parser.add_argument("--layer", type=int, default=41,
                    help="Model layer to extract features from")
parser.add_argument("--train_dir", type=str, required=True,
                    help="Path to training dataset directory")
parser.add_argument("--test_dir", type=str, required=True,
                    help="Path to test dataset directory")
parser.add_argument("--train_response_df", type=str, required=True,
                    help="CSV with train responses (id, answer)")
parser.add_argument("--test_response_df", type=str, required=True,
                    help="CSV with test responses (id, answer)")
parser.add_argument("--batch_size", type=int, default=128,
                    help="Training batch size")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="Maximum number of training epochs")
parser.add_argument("--learning_rate", type=float, default=1e-3,
                    help="Learning rate")
parser.add_argument("--n_runs", type=int, default=5,
                    help="Number of runs per training size for min/max/mean")
parser.add_argument("--random_seed", type=int, default=42,
                    help="Base random seed for reproducibility")
parser.add_argument("--ece_bins", type=int, default=10,
                    help="Number of uniform bins for ECE computation")
parser.add_argument("--output", type=str, default=None,
                    help="Output JSON file path")
parser.add_argument("--stack", action="store_true",
                    help="Whether to stack features before training")
parser.add_argument("--train_sizes", type=int, nargs='+',
                    default=[50, 100, 200, 500, 1000, 2500, 5000],
                    help="Training sizes to evaluate")

args = parser.parse_args()


# ── Helper Functions ────────────────────────────────────────────────────────
def mlp(input_dim, num_classes, hidden_dim=512):
    """Create MLP model architecture."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )


def ece_uniform(correct, p_hat, bins):
    """Histogram ECE with uniform bin edges."""
    inds = np.digitize(p_hat, bins) - 1
    ece_sum, n = 0.0, len(p_hat)
    for b in range(len(bins)-1):
        m = (inds == b)
        nb = int(m.sum())
        if nb == 0:
            continue
        frac_pos = correct[m].mean()
        mean_pred = p_hat[m].mean()
        ece_sum += nb * abs(frac_pos - mean_pred)
    return ece_sum / n


def brier_binary(correct, p_hat):
    """Brier score on correctness with selected-class probability."""
    return np.mean((p_hat - correct)**2)


def stratified_sample(X_df, y_series, n_samples, random_state):
    """
    Stratified sampling to maintain class distribution.
    Returns sampled X and y maintaining the original class proportions.
    """
    if n_samples >= len(X_df):
        return X_df, y_series
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(y_series, return_counts=True)
    class_props = class_counts / len(y_series)
    
    # Calculate samples per class
    samples_per_class = np.round(class_props * n_samples).astype(int)
    
    # Adjust for rounding errors
    while samples_per_class.sum() < n_samples:
        idx = np.argmax(class_props * n_samples - samples_per_class)
        samples_per_class[idx] += 1
    while samples_per_class.sum() > n_samples:
        idx = np.argmax(samples_per_class - class_props * n_samples)
        samples_per_class[idx] -= 1
    
    # Sample from each class
    rng = np.random.RandomState(random_state)
    sampled_indices = []
    
    for cls, n_cls_samples in zip(unique_classes, samples_per_class):
        cls_indices = y_series[y_series == cls].index
        if n_cls_samples > len(cls_indices):
            n_cls_samples = len(cls_indices)
        sampled = rng.choice(cls_indices, size=n_cls_samples, replace=False)
        sampled_indices.extend(sampled)
    
    return X_df.loc[sampled_indices], y_series.loc[sampled_indices]


def train_and_evaluate(X_train_subset, y_train_subset, X_test_df, y_test,
                       meta, args, run_seed):
    """
    Train model on subset and evaluate on fixed test set.
    Returns accuracy, brier score, and ECE.
    """
    # Train model
    model, history = fit(
        X_train_subset,
        y_train_subset.values,
        model=mlp,
        meta=meta,
        num_classes=4,
        metric="loss",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=25,
        n_epochs=args.n_epochs,
        early_stopping=True,
        val_split=0.2,
        random_state=run_seed,
    )
    
    # Evaluate on test set
    device = next(model.parameters()).device
    X_test_np = X_test_df.values.astype(np.float32)
    
    probs, test_acc, _ = _evaluate(
        model,
        X_test_np,
        y_test.values,
        model_type="mlp",
        metric_fn=lambda y, p: accuracy_score(y, p.argmax(1)),
        ce_loss=None,
        device=device,
    )
    
    # Calculate metrics
    pred_labels = probs.argmax(1)
    p_selected = probs[np.arange(len(probs)), pred_labels]
    p_selected = np.clip(p_selected, 1e-12, 1 - 1e-12)
    is_correct = (pred_labels == y_test.values).astype(float)
    
    accuracy = float(is_correct.mean())
    brier = float(brier_binary(is_correct, p_selected))
    
    ECE_BINS = np.linspace(0.0, 1.0, args.ece_bins + 1)
    ece = float(ece_uniform(is_correct, p_selected, ECE_BINS))
    
    return accuracy, brier, ece


# ── Main Execution ──────────────────────────────────────────────────────────
def main():
    layer = args.layer
    train_dir = pathlib.Path(args.train_dir)
    test_dir = pathlib.Path(args.test_dir)
    
    print(f"Training Size Sweep Experiment")
    print(f"=" * 50)
    print(f"Layer: {layer}")
    print(f"Training sizes: {args.train_sizes}")
    print(f"Number of runs per size: {args.n_runs}")
    print()
    
    # ── Feature Extraction (if needed) ──────────────────────────────────────
    if args.stack:
        print("Stacking features...")
        train_npz = extract_token_vectors(
            str(train_dir),
            output_file=f"{train_dir}/condensed/features_rs_L{layer}.npz",
            layers=layer,
            parts="rs",
            n_jobs=os.cpu_count(),
        )
        print(f"✅ Stacked train features: {train_npz}")
        
        test_npz = extract_token_vectors(
            str(test_dir),
            output_file=f"{test_dir}/condensed/features_rs_L{layer}.npz",
            layers=layer,
            parts="rs",
            n_jobs=os.cpu_count(),
        )
        print(f"✅ Stacked test features: {test_npz}")
        print()
    
    # ── Load Full Training Data ─────────────────────────────────────────────
    print("Loading training data...")
    X_train_df, meta = load_npz_features(
        train_dir / "condensed" / f"features_rs_L{layer}.npz"
    )
    
    y_train = pd.read_csv(args.train_response_df, index_col="id")["answer"]
    y_train = y_train.map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
    
    # Align indices
    common_idx = X_train_df.index.intersection(y_train.index)
    X_train_df = X_train_df.loc[common_idx]
    y_train = y_train.loc[common_idx]
    print(f"Full train set: {X_train_df.shape}, labels: {y_train.shape}")
    
    # ── Load Test Data (Fixed) ───────────────────────────────────────────────
    print("Loading test data...")
    X_test_df, _ = load_npz_features(
        test_dir / "condensed" / f"features_rs_L{layer}.npz"
    )
    
    y_test = pd.read_csv(args.test_response_df, index_col="id")["answer"]
    y_test = y_test.map({'A': 0, 'B': 1, 'C': 2, 'D': 3})
    
    common_idx = X_test_df.index.intersection(y_test.index)
    X_test_df = X_test_df.loc[common_idx]
    y_test = y_test.loc[common_idx]
    print(f"Test set (fixed): {X_test_df.shape}, labels: {y_test.shape}")
    print()
    
    # ── Training Size Sweep ──────────────────────────────────────────────────
    results = {}
    
    # Filter training sizes to those <= available data
    valid_train_sizes = [s for s in args.train_sizes if s <= len(X_train_df)]
    if len(valid_train_sizes) < len(args.train_sizes):
        print(f"Note: Skipping sizes > {len(X_train_df)} (exceeds available data)")
        print(f"Using sizes: {valid_train_sizes}")
        print()
    
    for train_size in valid_train_sizes:
        print(f"\n{'='*50}")
        print(f"Training Size: {train_size}")
        print(f"{'='*50}")
        
        acc_runs = []
        brier_runs = []
        ece_runs = []
        
        for run_idx in range(args.n_runs):
            run_seed = args.random_seed + run_idx * 1000 + train_size
            print(f"\n  Run {run_idx + 1}/{args.n_runs} (seed={run_seed})")
            
            # Sample training subset
            X_train_subset, y_train_subset = stratified_sample(
                X_train_df, y_train, train_size, run_seed
            )
            
            print(f"    Sampled {len(X_train_subset)} training examples")
            
            # Train and evaluate
            try:
                accuracy, brier, ece = train_and_evaluate(
                    X_train_subset, y_train_subset,
                    X_test_df, y_test,
                    meta, args, run_seed
                )
                
                acc_runs.append(accuracy)
                brier_runs.append(brier)
                ece_runs.append(ece)
                
                print(f"    Accuracy: {accuracy:.4f}")
                print(f"    Brier:    {brier:.4f}")
                print(f"    ECE:      {ece:.4f}")
                
            except Exception as e:
                print(f"    Error in run {run_idx + 1}: {e}")
                continue
        
        # Calculate statistics
        if acc_runs:
            results[train_size] = {
                "accuracy": {
                    "mean": float(np.mean(acc_runs)),
                    "min": float(np.min(acc_runs)),
                    "max": float(np.max(acc_runs)),
                    "std": float(np.std(acc_runs))
                },
                "brier": {
                    "mean": float(np.mean(brier_runs)),
                    "min": float(np.min(brier_runs)),
                    "max": float(np.max(brier_runs)),
                    "std": float(np.std(brier_runs))
                },
                "ece": {
                    "mean": float(np.mean(ece_runs)),
                    "min": float(np.min(ece_runs)),
                    "max": float(np.max(ece_runs)),
                    "std": float(np.std(ece_runs))
                },
                "n_runs_completed": len(acc_runs)
            }
            
            print(f"\n  Summary for size {train_size}:")
            print(f"    Accuracy: {results[train_size]['accuracy']['mean']:.4f} "
                  f"(min={results[train_size]['accuracy']['min']:.4f}, "
                  f"max={results[train_size]['accuracy']['max']:.4f})")
            print(f"    Brier:    {results[train_size]['brier']['mean']:.4f} "
                  f"(min={results[train_size]['brier']['min']:.4f}, "
                  f"max={results[train_size]['brier']['max']:.4f})")
            print(f"    ECE:      {results[train_size]['ece']['mean']:.4f} "
                  f"(min={results[train_size]['ece']['min']:.4f}, "
                  f"max={results[train_size]['ece']['max']:.4f})")
    
    # ── Save Results ─────────────────────────────────────────────────────────
    # Determine output path
    if args.output:
        json_path = pathlib.Path(args.output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = train_dir / f"train_size_sweep_{timestamp}.json"
    
    # Prepare metadata
    args_serializable = {
        k: (str(v) if isinstance(v, pathlib.Path) else v)
        for k, v in vars(args).items()
    }
    
    output_data = {
        "experiment": "training_size_impact",
        "timestamp": datetime.now().isoformat(),
        "args": args_serializable,
        "dataset_info": {
            "full_train_size": len(X_train_df),
            "test_size": len(X_test_df),
            "n_features": X_train_df.shape[1],
            "n_classes": 4
        },
        "results": results
    }
    
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Results saved to: {json_path}")
    
    # ── Generate Summary Report ──────────────────────────────────────────────
    report_path = json_path.with_suffix(".txt")
    with open(report_path, "w") as f:
        f.write("Training Size Impact Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Layer: {layer}\n")
        f.write(f"Test Set Size: {len(X_test_df)}\n")
        f.write(f"Number of Runs per Size: {args.n_runs}\n\n")
        
        f.write("Model Configuration:\n")
        f.write(f"  - Batch Size: {args.batch_size}\n")
        f.write(f"  - Learning Rate: {args.learning_rate}\n")
        f.write(f"  - Max Epochs: {args.n_epochs}\n")
        f.write(f"  - Hidden Dim: 512\n\n")
        
        f.write("Results by Training Size:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Size':<8} {'Metric':<10} {'Mean':<8} {'Min':<8} {'Max':<8} {'Std':<8}\n")
        f.write("-" * 60 + "\n")
        
        for size in sorted(results.keys()):
            for metric in ['accuracy', 'brier', 'ece']:
                f.write(f"{size:<8} {metric:<10} "
                       f"{results[size][metric]['mean']:<8.4f} "
                       f"{results[size][metric]['min']:<8.4f} "
                       f"{results[size][metric]['max']:<8.4f} "
                       f"{results[size][metric]['std']:<8.4f}\n")
            f.write("\n")
    
    print(f"Summary report saved to: {report_path}")
    
    # ── Print Final Summary ──────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*50}")
    print("\nAccuracy by Training Size:")
    for size in sorted(results.keys()):
        acc = results[size]['accuracy']
        print(f"  {size:5d}: {acc['mean']:.4f} ± {acc['std']:.4f} "
              f"[{acc['min']:.4f}, {acc['max']:.4f}]")


if __name__ == "__main__":
    main()