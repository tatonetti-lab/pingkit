#!/usr/bin/env python
"""
probe_train.py
==============
Train a probe on combined embedding data with configurable class balancing,
hold out a fraction for evaluation, report metrics with bootstrap CIs, and
save the model.
----------------------------------------------------------------------------
Example
-------
$ python probe_train.py \
      --data_dirs       /path/to/train /path/to/test \
      --labels_csvs     train_labels.csv test_labels.csv \
      --label_col       correct \
      --parts           rs \
      --layers          37 \
      --model_type      mlp \
      --restack         true \
      --train_class_sizes 100 900 \
      --holdout_frac    0.05 \
      --output          my_model
----------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------- pingkit -------------------------------- #
from pingkit.extraction import extract_token_vectors
from pingkit.model import fit, load_npz_features, save_artifacts, _evaluate


# --------------------------- cli helpers ------------------------------ #
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(
        description="Train probe on combined data with optional class balancing and holdout evaluation."
    )
    p.add_argument(
        "--data_dirs",
        required=True,
        nargs="+",
        type=Path,
        help="Directories containing embeddings (rs/, attn/, mlp/ folders). Can specify multiple.",
    )
    p.add_argument(
        "--labels_csvs",
        required=True,
        nargs="+",
        type=Path,
        help="CSVs with sample IDs and labels (one per data_dir, in same order).",
    )
    p.add_argument(
        "--label_col",
        default="label",
        help="Column name containing ground-truth labels.",
    )
    p.add_argument(
        "--id_col",
        default="id",
        help="Column in labels CSV that holds sample IDs (must match embedding filenames).",
    )
    p.add_argument(
        "--parts",
        default="rs",
        help='Comma-separated list of embedding parts to use (e.g. "rs", "rs,attn").',
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[41],
        help="Layer(s) to extract features from. Can specify multiple (e.g. --layers 20 30 40).",
    )
    p.add_argument(
        "--model_type",
        choices=["mlp", "cnn"],
        default="mlp",
        help="Probe architecture to train.",
    )
    p.add_argument(
        "--restack",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to re-stack embeddings into NPZ files (set false to use existing).",
    )
    p.add_argument(
        "--train_class_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Number of samples per class for training (e.g. --train_class_sizes 100 900 for class 0=100, class 1=900). "
             "If not specified, uses all available samples. Samples are drawn from post-holdout data.",
    )
    p.add_argument(
        "--holdout_frac",
        type=float,
        default=0.05,
        help="Fraction of data to hold out for evaluation (only used when --train_class_sizes is not specified).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for artifacts. Auto-generates if not provided.",
    )
    p.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Maximum epochs for training.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Mini-batch size for training.",
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    p.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation during training.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience.",
    )
    p.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for holdout split and model init.",
    )
    p.add_argument(
        "--bootstrap_B",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for CI estimation.",
    )
    p.add_argument(
        "--bootstrap_seed",
        type=int,
        default=8675309,
        help="Random seed for bootstrap resampling.",
    )
    p.add_argument(
        "--ece_bins",
        type=int,
        default=10,
        help="Number of uniform bins for ECE computation.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Torch device string ("cuda" or "cpu").',
    )
    return p.parse_args(argv)


# -------------------------- utility funcs ----------------------------- #
def compute_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    Compute ROC AUC:
      - binary: use positive class score (column 1)
      - multiclass: macro One-vs-Rest
    Returns float('nan') if AUC cannot be computed.
    """
    try:
        n_classes = probs.shape[1]
        if n_classes == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def ece_uniform(correct: np.ndarray, p_hat: np.ndarray, bins: np.ndarray) -> float:
    """Compute histogram ECE with uniform bin edges."""
    inds = np.digitize(p_hat, bins) - 1
    ece_sum, n = 0.0, len(p_hat)
    for b in range(len(bins) - 1):
        mask = inds == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        frac_pos = correct[mask].mean()
        mean_pred = p_hat[mask].mean()
        ece_sum += nb * abs(frac_pos - mean_pred)
    return ece_sum / n


def brier_binary(correct: np.ndarray, p_hat: np.ndarray) -> float:
    """Brier score on correctness-as-label with selected-class probability."""
    return float(np.mean((p_hat - correct) ** 2))


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute percentile confidence interval."""
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def layer_str(layers: List[int]) -> str:
    """Create consistent layer string for file naming."""
    return "_".join(map(str, sorted(layers)))


def stack_features(
    embedding_dir: Path,
    layers: List[int],
    parts: List[str],
    n_jobs: int = 8,
) -> Path:
    """Stack embeddings for given layers/parts into a single NPZ and return its path.

    When called (i.e., --restack true), this overwrites any existing matching NPZ.
    """
    lstr = layer_str(layers)
    out = embedding_dir / "condensed" / f"features_{'-'.join(parts)}_L{lstr}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Overwrite behavior when restack is true: delete existing NPZ with same name.
    if out.exists():
        out.unlink()

    extract_token_vectors(
        embedding_dir=str(embedding_dir),
        parts=parts,
        layers=layers,
        output_file=str(out.with_suffix("")),
        save_csv=False,
        n_jobs=n_jobs,
    )
    return out



def load_labels(
    labels_path: Path,
    id_col: str,
    label_col: str,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Load labels from CSV (without encoding). Returns (label_series, full_dataframe)."""
    df = pd.read_csv(labels_path)
    if id_col not in df.columns or label_col not in df.columns:
        sys.exit(f"{labels_path} must contain columns [{id_col}, {label_col}]")
    df = df.set_index(id_col)
    return df[label_col], df


def align_data(
    X_df: pd.DataFrame,
    y: pd.Series,
    name: str = "data",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Align features and labels by index, warn about missing samples."""
    common = X_df.index.intersection(y.index)
    if len(common) < len(y):
        print(
            f"  Warning: {len(y) - len(common)} {name} samples missing embeddings, "
            "dropping."
        )
    return X_df.loc[common], y.loc[common]


def run_bootstrap(
    y_true: np.ndarray,
    pred: np.ndarray,
    conf: np.ndarray,
    probs: np.ndarray,
    ece_bins: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Run stratified bootstrap to compute CIs for accuracy, AUC, Brier, and ECE."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    idx_by_c = {c: np.where(y_true == c)[0] for c in classes}
    n_by_c = {c: len(idx_by_c[c]) for c in classes}

    acc_samples = np.empty(n_bootstrap, dtype=np.float64)
    auc_samples = np.empty(n_bootstrap, dtype=np.float64)
    brier_samples = np.empty(n_bootstrap, dtype=np.float64)
    ece_samples = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx_boot = np.concatenate(
            [rng.choice(idx_by_c[c], size=n_by_c[c], replace=True) for c in classes]
        )
        yb = y_true[idx_boot]
        pb = pred[idx_boot]
        cb = conf[idx_boot]
        probs_b = probs[idx_boot]
        corr = (pb == yb).astype(float)

        acc_samples[b] = corr.mean()
        auc_samples[b] = compute_auc(yb, probs_b)
        brier_samples[b] = brier_binary(corr, cb)
        ece_samples[b] = ece_uniform(corr, cb, ece_bins)

    return {
        "accuracy": {"samples": acc_samples, "ci": percentile_ci(acc_samples)},
        "auc": {"samples": auc_samples, "ci": percentile_ci(auc_samples[~np.isnan(auc_samples)])},
        "brier": {"samples": brier_samples, "ci": percentile_ci(brier_samples)},
        "ece": {"samples": ece_samples, "ci": percentile_ci(ece_samples)},
    }


def subsample_by_class(
    X: pd.DataFrame,
    y: pd.Series,
    class_sizes: List[int],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Subsample data to have specified number of samples per class.
    
    Args:
        X: Feature dataframe
        y: Label series (encoded as integers 0, 1, 2, ...)
        class_sizes: List of sample counts, where index corresponds to class label
        random_state: Random seed for reproducibility
    
    Returns:
        Subsampled (X, y) tuple
    """
    rng = np.random.default_rng(random_state)
    classes = sorted(np.unique(y))
    
    if len(class_sizes) != len(classes):
        sys.exit(
            f"--train_class_sizes must have {len(classes)} values (one per class), "
            f"got {len(class_sizes)}"
        )
    
    selected_indices = []
    for cls, n_samples in zip(classes, class_sizes):
        cls_indices = y[y == cls].index.tolist()
        available = len(cls_indices)
        
        if n_samples > available:
            print(
                f"  Warning: Requested {n_samples} samples for class {cls}, "
                f"but only {available} available. Using all."
            )
            selected_indices.extend(cls_indices)
        else:
            selected = rng.choice(cls_indices, size=n_samples, replace=False)
            selected_indices.extend(selected)
    
    return X.loc[selected_indices], y.loc[selected_indices]


# ------------------------------- main --------------------------------- #
def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    layers = args.layers
    lstr = layer_str(layers)
    ece_bins = np.linspace(0.0, 1.0, args.ece_bins + 1)

    # Validate inputs
    if len(args.data_dirs) != len(args.labels_csvs):
        sys.exit("Must provide same number of --data_dirs and --labels_csvs")

    print(f"Layers: {layers}")
    print(f"Parts: {parts}")
    print(f"Device: {args.device}")
    print(f"Data sources: {len(args.data_dirs)}")

    # ── Feature extraction and loading ──
    all_X = []
    all_y = []
    all_label_dfs = []
    meta = None

    for i, (data_dir, labels_csv) in enumerate(zip(args.data_dirs, args.labels_csvs)):
        print(f"\nLoading data source {i + 1}: {data_dir}")

        parts_str = "-".join(parts)
        npz_path = data_dir / "condensed" / f"features_{parts_str}_L{lstr}.npz"

        if args.restack.lower() == "true":
            print("  Stacking features...")
            npz_path = stack_features(
                embedding_dir=data_dir,
                layers=layers,
                parts=parts,
                n_jobs=os.cpu_count() or 8,
            )
            print(f"  NPZ: {npz_path}")
        else:
            if not npz_path.exists():
                sys.exit(f"{npz_path} missing; rerun with --restack true first.")

        X_df, m = load_npz_features(str(npz_path))
        if meta is None:
            meta = m

        y, label_df = load_labels(labels_csv, args.id_col, args.label_col)
        X_df, y = align_data(X_df, y, f"source {i + 1}")
        # Also align the full label dataframe
        label_df = label_df.loc[X_df.index]

        print(f"  Loaded: {X_df.shape[0]} samples, {X_df.shape[1]} features")

        all_X.append(X_df)
        all_y.append(y)
        all_label_dfs.append(label_df)

    # ── Combine all data ──
    print("\nCombining datasets...")
    X_combined = pd.concat(all_X, axis=0)
    y_combined = pd.concat(all_y, axis=0)
    labels_combined = pd.concat(all_label_dfs, axis=0)

    # Handle duplicate indices (samples present in multiple sources)
    if X_combined.index.duplicated().any():
        print(f"  Warning: {X_combined.index.duplicated().sum()} duplicate sample IDs found, keeping first occurrence.")
        X_combined = X_combined[~X_combined.index.duplicated(keep='first')]
        y_combined = y_combined[~y_combined.index.duplicated(keep='first')]
        labels_combined = labels_combined[~labels_combined.index.duplicated(keep='first')]

    print(f"Combined: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")

    # ── Encode labels ──
    encoder = None
    if y_combined.dtype == "object" or not np.issubdtype(y_combined.dtype, np.number):
        encoder = LabelEncoder()
        y_combined = pd.Series(
            encoder.fit_transform(y_combined),
            index=y_combined.index,
            name=y_combined.name,
        )
        print(f"Label mapping: {dict(enumerate(encoder.classes_))}")

    num_classes = len(np.unique(y_combined))
    print(f"Classes: {num_classes}")

    # ── Split into train and holdout ──
    if args.train_class_sizes is not None:
        # Holdout = everything not selected for training
        print(f"\nSubsampling training data by class: {args.train_class_sizes}")
        X_train, y_train = subsample_by_class(
            X_combined,
            y_combined,
            args.train_class_sizes,
            random_state=args.random_state,
        )
        
        # Holdout is everything else
        holdout_idx = X_combined.index.difference(X_train.index)
        X_holdout = X_combined.loc[holdout_idx]
        y_holdout = y_combined.loc[holdout_idx]
        labels_holdout = labels_combined.loc[holdout_idx]
        
        print(f"\n  Train: {X_train.shape[0]} samples")
        for cls in sorted(np.unique(y_train)):
            cls_count = (y_train == cls).sum()
            cls_pct = 100 * cls_count / len(y_train)
            print(f"    Class {cls}: {cls_count:,} ({cls_pct:.1f}%)")
        
        print(f"\n  Holdout: {X_holdout.shape[0]} samples (all remaining)")
        for cls in sorted(np.unique(y_holdout)):
            cls_count = (y_holdout == cls).sum()
            cls_pct = 100 * cls_count / len(y_holdout)
            print(f"    Class {cls}: {cls_count:,} ({cls_pct:.1f}%)")
    else:
        # Standard stratified split
        print(f"\nSplitting data (holdout={args.holdout_frac:.1%})...")
        train_idx, holdout_idx = train_test_split(
            X_combined.index,
            test_size=args.holdout_frac,
            stratify=y_combined,
            random_state=args.random_state,
        )
        
        X_train = X_combined.loc[train_idx]
        X_holdout = X_combined.loc[holdout_idx]
        y_train = y_combined.loc[train_idx]
        y_holdout = y_combined.loc[holdout_idx]
        labels_holdout = labels_combined.loc[holdout_idx]
        
        print(f"\n  Train: {X_train.shape[0]} samples")
        for cls in sorted(np.unique(y_train)):
            cls_count = (y_train == cls).sum()
            cls_pct = 100 * cls_count / len(y_train)
            print(f"    Class {cls}: {cls_count:,} ({cls_pct:.1f}%)")
        
        print(f"\n  Holdout: {X_holdout.shape[0]} samples ({args.holdout_frac:.1%} stratified)")
        for cls in sorted(np.unique(y_holdout)):
            cls_count = (y_holdout == cls).sum()
            cls_pct = 100 * cls_count / len(y_holdout)
            print(f"    Class {cls}: {cls_count:,} ({cls_pct:.1f}%)")

    # ── Train model ──
    print("\nTraining model...")
    model, history = fit(
        X_train,
        y_train.values,
        model_type=args.model_type,
        meta=meta if args.model_type == "cnn" else None,
        num_classes=num_classes,
        metric="loss",
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        n_epochs=args.n_epochs,
        early_stopping=True,
        val_split=args.val_split,
        random_state=args.random_state,
        device=args.device,
    )

    # ── Evaluate on holdout set ──
    print("\nEvaluating on holdout set...")
    device = next(model.parameters()).device
    X_holdout_np = X_holdout.values.astype(np.float32)

    probs, holdout_acc, _ = _evaluate(
        model,
        X_holdout_np,
        y_holdout.values,
        model_type=args.model_type,
        metric_fn=lambda y, p: accuracy_score(y, p.argmax(1)),
        ce_loss=None,
        device=device,
    )

    pred_labels = probs.argmax(1)
    p_selected = np.clip(probs[np.arange(len(probs)), pred_labels], 1e-12, 1 - 1e-12)
    is_correct = (pred_labels == y_holdout.values).astype(int)

    # Point estimates
    acc_point = float(is_correct.mean())
    auc_point = compute_auc(y_holdout.values, probs)
    brier_point = brier_binary(is_correct.astype(float), p_selected)
    ece_point = ece_uniform(is_correct.astype(float), p_selected, ece_bins)

    # Compute per-class accuracy first (needed for macro avg)
    y_holdout_arr = y_holdout.values
    class_accuracies = []
    per_class_print = []
    per_class_metrics = {}
    for cls in sorted(np.unique(y_holdout_arr)):
        cls_mask = y_holdout_arr == cls
        cls_correct = int(is_correct[cls_mask].sum())
        cls_total = int(cls_mask.sum())
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
        # Average prediction (prob of class 1 for binary, or mean predicted class for multiclass)
        if num_classes == 2:
            avg_pred = float(probs[cls_mask, 1].mean()) if cls_total > 0 else 0.0
        else:
            avg_pred = float(pred_labels[cls_mask].mean()) if cls_total > 0 else 0.0
        class_accuracies.append(cls_acc)
        per_class_print.append((cls, cls_acc, cls_correct, cls_total, avg_pred))
        per_class_metrics[int(cls)] = {
            "accuracy": float(cls_acc),
            "correct": cls_correct,
            "total": cls_total,
            "avg_pred": avg_pred,
        }
    macro_acc = float(np.mean(class_accuracies))
    per_class_metrics["macro_avg"] = macro_acc

    print(f"\n  Overall Metrics:")
    print(f"    Accuracy:       {acc_point:.4f}")
    print(f"    Macro Accuracy: {macro_acc:.4f}")
    print(f"    AUC:            {auc_point:.4f}")
    print(f"    Brier:          {brier_point:.4f}")
    print(f"    ECE:            {ece_point:.4f}")
    
    print(f"\n  Per-Class Metrics:")
    print(f"    {'Class':<8} {'Acc':>8} {'Correct':>10} {'AvgPred':>10}")
    print(f"    {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for cls, cls_acc, cls_correct, cls_total, avg_pred in per_class_print:
        print(f"    {cls:<8} {cls_acc:>8.4f} {cls_correct:>5}/{cls_total:<4} {avg_pred:>10.4f}")
    print(f"    {'Macro':<8} {macro_acc:>8.4f}")

    # ── Bootstrap CIs ──
    print(f"\nRunning bootstrap (B={args.bootstrap_B})...")
    bootstrap_results = run_bootstrap(
        y_true=y_holdout.values,
        pred=pred_labels,
        conf=p_selected,
        probs=probs,
        ece_bins=ece_bins,
        n_bootstrap=args.bootstrap_B,
        seed=args.bootstrap_seed,
    )

    acc_ci = bootstrap_results["accuracy"]["ci"]
    auc_ci = bootstrap_results["auc"]["ci"]
    brier_ci = bootstrap_results["brier"]["ci"]
    ece_ci = bootstrap_results["ece"]["ci"]

    print(f"  Accuracy 95% CI: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
    print(f"  AUC 95% CI:      [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
    print(f"  Brier 95% CI:    [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]")
    print(f"  ECE 95% CI:      [{ece_ci[0]:.4f}, {ece_ci[1]:.4f}]")

    # ── Save artifacts ──
    if args.output is not None:
        artifact_root = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_root = (
            args.data_dirs[0] / "artifacts" / args.model_type.upper() / f"L{lstr}_{timestamp}"
        )

    artifact_root.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving artifacts to: {artifact_root}")

    # Save model
    save_artifacts(model, path=str(artifact_root), meta=meta)

    # Save label encoder if used
    if encoder is not None:
        encoder_mapping = {int(i): str(c) for i, c in enumerate(encoder.classes_)}
        with open(artifact_root / "label_encoder.json", "w") as f:
            json.dump(encoder_mapping, f, indent=2)

    # Save metrics JSON
    args_serializable = {
        k: ([str(p) for p in v] if isinstance(v, list) and v and isinstance(v[0], Path) else (str(v) if isinstance(v, Path) else v))
        for k, v in vars(args).items()
    }
    metrics_json = {
        "args": args_serializable,
        "layers": layers,
        "point": {"accuracy": acc_point, "macro_accuracy": macro_acc, "auc": auc_point, "brier": brier_point, "ece": ece_point},
        "per_class": per_class_metrics,
        "ci_95": {
            "accuracy": {"low": acc_ci[0], "high": acc_ci[1]},
            "auc": {"low": auc_ci[0], "high": auc_ci[1]},
            "brier": {"low": brier_ci[0], "high": brier_ci[1]},
            "ece": {"low": ece_ci[0], "high": ece_ci[1]},
        },
        "bootstrap": {
            "B": args.bootstrap_B,
            "seed": args.bootstrap_seed,
            "ece_bins": ece_bins.tolist(),
        },
        "dataset": {
            "total_samples": int(X_combined.shape[0]),
            "train_samples": int(X_train.shape[0]),
            "holdout_samples": int(X_holdout.shape[0]),
            "num_features": int(X_combined.shape[1]),
            "num_classes": num_classes,
            "holdout_frac": args.holdout_frac,
            "train_class_sizes": args.train_class_sizes,
        },
    }

    with open(artifact_root / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    # Save holdout predictions (includes all original label columns)
    predictions_df = labels_holdout.copy()
    predictions_df["pred_label"] = pred_labels
    predictions_df["confidence"] = p_selected
    predictions_df["is_correct"] = is_correct
    # Add probability columns dynamically
    for i in range(num_classes):
        predictions_df[f"prob_{i}"] = probs[:, i]
    
    # Reset index to make id a column
    predictions_df = predictions_df.reset_index()

    predictions_df.to_csv(artifact_root / "holdout_predictions.csv", index=False)

    # Save human-readable results
    if args.train_class_sizes:
        class_balance_str = f"  - Train class sizes: {args.train_class_sizes}"
        holdout_str = f"  - Holdout samples: {X_holdout.shape[0]} (remaining after class selection)"
    else:
        class_balance_str = "  - Train class sizes: all available"
        holdout_str = f"  - Holdout samples: {X_holdout.shape[0]} ({args.holdout_frac:.1%} stratified)"
    
    # Build per-class accuracy string for results text
    per_class_str = "Per-Class Metrics:\n"
    per_class_str += f"  {'Class':<8} {'Acc':>8} {'Correct':>10} {'AvgPred':>10}\n"
    for cls, metrics in per_class_metrics.items():
        if cls == "macro_avg":
            continue
        per_class_str += f"  {cls:<8} {metrics['accuracy']:>8.4f} {metrics['correct']:>5}/{metrics['total']:<4} {metrics['avg_pred']:>10.4f}\n"
    per_class_str += f"  {'Macro':<8} {per_class_metrics['macro_avg']:>8.4f}\n"
    
    results_text = f"""Probe Training Results
======================

Layers: {layers}
Parts: {parts}

Dataset:
  - Total samples:   {X_combined.shape[0]}
  - Train samples:   {X_train.shape[0]}
{holdout_str}
  - Features:        {X_combined.shape[1]}
  - Classes:         {num_classes}
{class_balance_str}

Holdout Set Point Estimates:
  - Accuracy:       {acc_point:.4f}
  - Macro Accuracy: {macro_acc:.4f}
  - AUC:            {auc_point:.4f}
  - Brier:          {brier_point:.4f}
  - ECE:            {ece_point:.4f}

{per_class_str}
95% Bootstrap CIs (B={args.bootstrap_B}, seed={args.bootstrap_seed}):
  - Accuracy: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]
  - AUC:      [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]
  - Brier:    [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]
  - ECE:      [{ece_ci[0]:.4f}, {ece_ci[1]:.4f}]

Model Configuration:
  - Type:          {args.model_type}
  - Batch Size:    {args.batch_size}
  - Learning Rate: {args.learning_rate}
  - Max Epochs:    {args.n_epochs}
  - Val Split:     {args.val_split}
  - Patience:      {args.patience}
  - Random State:  {args.random_state}

Data Sources:
"""
    for i, (d, l) in enumerate(zip(args.data_dirs, args.labels_csvs)):
        results_text += f"  {i + 1}. {d} / {l}\n"

    with open(artifact_root / "results.txt", "w") as f:
        f.write(results_text)

    print(f"\nResults saved to: {artifact_root}")
    print(
        f"\nSummary: ACC {acc_point:.4f} [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}] | "
        f"Macro ACC {macro_acc:.4f} | "
        f"AUC {auc_point:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}] | "
        f"Brier {brier_point:.4f} [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}] | "
        f"ECE {ece_point:.4f} [{ece_ci[0]:.4f}, {ece_ci[1]:.4f}]"
    )


if __name__ == "__main__":
    main()