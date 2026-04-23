#!/usr/bin/env python
"""
Run batch inference on pre-embedded examples using a trained probe.
Loads features from an embedding directory, runs the probe, evaluates
against ground-truth labels, and saves predictions to CSV.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from pingkit.extraction import extract_token_vectors
from pingkit.model import load_artifacts, load_npz_features, _evaluate


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch probe inference and evaluation.")
    p.add_argument("--data_dir", type=Path, required=True,
                   help="Directory containing embeddings.")
    p.add_argument("--labels_csv", type=Path, required=True,
                   help="CSV with sample IDs and labels.")
    p.add_argument("--label_col", default="correct",
                   help="Column name containing ground-truth labels.")
    p.add_argument("--id_col", default="id",
                   help="Column in labels CSV that holds sample IDs.")
    p.add_argument("--artifact_dir", type=Path, required=True,
                   help="Directory containing trained probe artifacts.")
    p.add_argument("--layers", type=int, nargs="+", default=[37],
                   help="Layer(s) to extract features from.")
    p.add_argument("--parts", default="rs",
                   help='Comma-separated embedding parts (e.g. "rs", "rs,attn").')
    p.add_argument("--restack", type=str, default="true", choices=["true", "false"],
                   help="Whether to re-stack embeddings into NPZ files.")
    p.add_argument("--output_csv", type=Path, default=None,
                   help="Output CSV path for predictions. Defaults to <data_dir>/predictions.csv.")
    p.add_argument("--bootstrap_B", type=int, default=1000,
                   help="Number of bootstrap resamples for CI estimation.")
    p.add_argument("--bootstrap_seed", type=int, default=8675309,
                   help="Random seed for bootstrap resampling.")
    p.add_argument("--ece_bins", type=int, default=10,
                   help="Number of uniform bins for ECE computation and bin stats.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help='Torch device string.')
    return p.parse_args(argv)


# ── Metrics ──

def compute_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    try:
        n_classes = probs.shape[1]
        if n_classes == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def ece_uniform(correct: np.ndarray, p_hat: np.ndarray, bins: np.ndarray) -> float:
    """
    Uniform-bin Expected Calibration Error using top-1 confidence p_hat and
    correctness indicator correct in {0,1}.
    """
    p_hat = np.clip(p_hat, 0.0, 1.0 - 1e-12)
    inds = np.digitize(p_hat, bins) - 1
    inds = np.clip(inds, 0, len(bins) - 2)

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
    return float(np.mean((p_hat - correct) ** 2))


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def run_bootstrap(
    y_true: np.ndarray, pred: np.ndarray, conf: np.ndarray,
    probs: np.ndarray, ece_bins: np.ndarray, n_bootstrap: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    idx_by_c = {c: np.where(y_true == c)[0] for c in classes}
    n_by_c = {c: len(idx_by_c[c]) for c in classes}

    acc_s = np.empty(n_bootstrap)
    auc_s = np.empty(n_bootstrap)
    brier_s = np.empty(n_bootstrap)
    ece_s = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = np.concatenate(
            [rng.choice(idx_by_c[c], size=n_by_c[c], replace=True) for c in classes]
        )
        yb, pb, cb, probs_b = y_true[idx], pred[idx], conf[idx], probs[idx]
        corr = (pb == yb).astype(float)
        acc_s[b] = corr.mean()
        auc_s[b] = compute_auc(yb, probs_b)
        brier_s[b] = brier_binary(corr, cb)
        ece_s[b] = ece_uniform(corr, cb, ece_bins)

    return {
        "accuracy": percentile_ci(acc_s),
        "auc": percentile_ci(auc_s[~np.isnan(auc_s)]),
        "brier": percentile_ci(brier_s),
        "ece": percentile_ci(ece_s),
    }


def calibration_bin_stats(
    correct: np.ndarray,
    conf: np.ndarray,
    bins: np.ndarray,
) -> pd.DataFrame:
    """
    Bin-by-bin calibration stats for top-1 confidence.

    Returns a DataFrame with:
      bin_lo, bin_hi, n, frac, acc, avg_conf, gap, abs_gap, ece_contrib
    """
    correct = correct.astype(float)
    conf = np.clip(conf.astype(float), 0.0, 1.0 - 1e-12)

    n_total = len(conf)
    inds = np.digitize(conf, bins) - 1
    inds = np.clip(inds, 0, len(bins) - 2)

    rows = []
    for b in range(len(bins) - 1):
        lo, hi = float(bins[b]), float(bins[b + 1])
        mask = inds == b
        nb = int(mask.sum())
        frac = nb / n_total if n_total > 0 else 0.0

        if nb > 0:
            acc = float(correct[mask].mean())
            avg_conf = float(conf[mask].mean())
            gap = acc - avg_conf
            abs_gap = abs(gap)
            ece_contrib = frac * abs_gap
        else:
            acc = float("nan")
            avg_conf = float("nan")
            gap = float("nan")
            abs_gap = float("nan")
            ece_contrib = 0.0

        rows.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "n": nb,
            "frac": frac,
            "acc": acc,
            "avg_conf": avg_conf,
            "gap": gap,
            "abs_gap": abs_gap,
            "ece_contrib": ece_contrib,
        })

    return pd.DataFrame(rows)


def layer_str(layers: List[int]) -> str:
    return "_".join(map(str, sorted(layers)))


def stack_features(embedding_dir: Path, layers: List[int], parts: List[str]) -> Path:
    lstr = layer_str(layers)
    out = embedding_dir / "condensed" / f"features_{'-'.join(parts)}_L{lstr}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    extract_token_vectors(
        embedding_dir=str(embedding_dir),
        parts=parts,
        layers=layers,
        output_file=str(out.with_suffix("")),
        save_csv=False,
        n_jobs=os.cpu_count() or 8,
    )
    return out


# ── Main ──

def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    layers = args.layers
    lstr = layer_str(layers)

    print(f"Layers: {layers}")
    print(f"Parts: {parts}")
    print(f"Device: {args.device}")
    print(f"Artifact dir: {args.artifact_dir}")

    # ── Load / stack features ──
    parts_str = "-".join(parts)
    npz_path = args.data_dir / "condensed" / f"features_{parts_str}_L{lstr}.npz"

    if args.restack.lower() == "true":
        print("\nStacking features...")
        npz_path = stack_features(args.data_dir, layers, parts)
        print(f"  NPZ: {npz_path}")
    else:
        if not npz_path.exists():
            sys.exit(f"{npz_path} missing; rerun with --restack true.")

    X_df, meta = load_npz_features(str(npz_path))

    # ── Load labels ──
    df_labels = pd.read_csv(args.labels_csv)
    if args.id_col not in df_labels.columns or args.label_col not in df_labels.columns:
        sys.exit(f"{args.labels_csv} must contain columns [{args.id_col}, {args.label_col}]")
    df_labels = df_labels.set_index(args.id_col)
    y_raw = df_labels[args.label_col]

    # Align
    common = X_df.index.intersection(y_raw.index)
    if len(common) < len(y_raw):
        print(f"  Warning: {len(y_raw) - len(common)} samples missing embeddings, dropping.")
    X_df = X_df.loc[common]
    y_raw = y_raw.loc[common]
    df_labels = df_labels.loc[common]

    print(f"Loaded {X_df.shape[0]} samples, {X_df.shape[1]} features")

    # ── Encode labels (match training encoder if available) ──
    label_encoder_path = args.artifact_dir / "label_encoder.json"
    inv_labels: Optional[dict] = None
    if label_encoder_path.exists():
        with open(label_encoder_path, "r") as f:
            label_map = json.load(f)
        inv_labels = {int(k): v for k, v in label_map.items()}
        fwd = {v: int(k) for k, v in label_map.items()}
        y = y_raw.map(fwd)
        if y.isna().any():
            unmapped = y_raw[y.isna()].unique().tolist()
            sys.exit(f"Labels not in encoder mapping: {unmapped}")
        y = y.astype(int)
        print(f"Label mapping (from encoder): {inv_labels}")
    else:
        if y_raw.dtype == "object" or not np.issubdtype(y_raw.dtype, np.number):
            encoder = LabelEncoder()
            y = pd.Series(encoder.fit_transform(y_raw), index=y_raw.index, name=y_raw.name)
            inv_labels = {i: str(c) for i, c in enumerate(encoder.classes_)}
            print(f"Label mapping (auto): {inv_labels}")
        else:
            y = y_raw.astype(int)

    num_classes = len(np.unique(y))
    print(f"Classes: {num_classes}")
    for cls in sorted(np.unique(y)):
        print(f"  Class {cls}: {(y == cls).sum()}")

    # ── ECE bins ──
    # For binary, top-1 confidence is always >= 0.5, so we bin over [0.5, 1.0]
    # to avoid guaranteed-empty bins below 0.5.
    if num_classes == 2:
        ece_bins = np.linspace(0.5, 1.0, args.ece_bins + 1)
    else:
        ece_bins = np.linspace(0.0, 1.0, args.ece_bins + 1)

    # ── Load probe ──
    print(f"\nLoading probe from {args.artifact_dir}...")
    model, probe_meta = load_artifacts(str(args.artifact_dir))
    device_str = args.device
    if device_str != "cpu":
        model = model.to(device_str)
    device = next(model.parameters()).device

    # ── Run inference ──
    print("Running inference...")
    X_np = X_df.values.astype(np.float32)

    probs, acc, _ = _evaluate(
        model,
        X_np,
        y.values,
        model_type="mlp",
        metric_fn=lambda yt, p: accuracy_score(yt, p.argmax(1)),
        ce_loss=None,
        device=device,
    )

    pred_labels = probs.argmax(1)
    p_selected = np.clip(probs[np.arange(len(probs)), pred_labels], 1e-12, 1 - 1e-12)
    is_correct = (pred_labels == y.values).astype(int)

    # ── Point estimates ──
    acc_point = float(is_correct.mean())
    auc_point = compute_auc(y.values, probs)
    brier_point = brier_binary(is_correct.astype(float), p_selected)
    ece_point = ece_uniform(is_correct.astype(float), p_selected, ece_bins)

    # Per-class
    class_accuracies = []
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\n  Overall:")
    print(f"    Accuracy: {acc_point:.4f}")

    print(f"\n  Per-Class:")
    print(f"    {'Class':<12} {'Acc':>8} {'Correct':>10} {'AvgPred':>10}")
    print(f"    {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
    for cls in sorted(np.unique(y.values)):
        mask = y.values == cls
        cls_correct = int(is_correct[mask].sum())
        cls_total = int(mask.sum())
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
        class_accuracies.append(cls_acc)
        if num_classes == 2:
            avg_pred = float(probs[mask, 1].mean())
        else:
            avg_pred = float(pred_labels[mask].mean())
        label_name = inv_labels[cls] if inv_labels and cls in inv_labels else str(cls)
        print(f"    {label_name:<12} {cls_acc:>8.4f} {cls_correct:>5}/{cls_total:<4} {avg_pred:>10.4f}")

    macro_acc = float(np.mean(class_accuracies))
    print(f"    {'Macro':<12} {macro_acc:>8.4f}")

    print(f"\n    AUC:   {auc_point:.4f}")
    print(f"    Brier: {brier_point:.4f}")
    print(f"    ECE:   {ece_point:.4f}")

    # ── ECE-relevant bin stats ──
    bin_df = calibration_bin_stats(is_correct.astype(float), p_selected, ece_bins)
    ece_from_bins = float(bin_df["ece_contrib"].sum())

    # Pretty-print
    if num_classes == 2:
        bin_range_str = "[0.5, 1.0]"
    else:
        bin_range_str = "[0.0, 1.0]"

    print(f"\n  Calibration by confidence bin ({args.ece_bins} bins over {bin_range_str}):")
    print(f"    {'Bin':<13} {'N':>7} {'Frac':>8} {'Acc':>8} {'AvgConf':>10} {'Gap':>10} {'|Gap|':>10} {'ECE contrib':>12}")
    print(f"    {'-'*13} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    for _, r in bin_df.iterrows():
        lo, hi = r["bin_lo"], r["bin_hi"]
        is_last = np.isclose(hi, float(ece_bins[-1]))
        bin_label = f"[{lo:.2f},{hi:.2f}{']' if is_last else ')'}"
        n = int(r["n"])
        frac = r["frac"]
        acc_b = r["acc"]
        avg_c = r["avg_conf"]
        gap = r["gap"]
        abs_gap = r["abs_gap"]
        contrib = r["ece_contrib"]

        def fmt(x: float) -> str:
            return "   nan  " if (isinstance(x, float) and np.isnan(x)) else f"{x:8.4f}"

        print(
            f"    {bin_label:<13} {n:7d} {frac:8.3f} "
            f"{fmt(acc_b)} {fmt(avg_c)} {fmt(gap)} {fmt(abs_gap)} {contrib:12.6f}"
        )

    print(f"\n    ECE check (sum of bin contribs): {ece_from_bins:.6f}")

    # ── Bootstrap CIs ──
    print(f"\n  Bootstrap CIs (B={args.bootstrap_B}):")
    ci = run_bootstrap(
        y.values, pred_labels, p_selected, probs,
        ece_bins, args.bootstrap_B, args.bootstrap_seed,
    )
    print(f"    Accuracy: [{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}]")
    print(f"    AUC:      [{ci['auc'][0]:.4f}, {ci['auc'][1]:.4f}]")
    print(f"    Brier:    [{ci['brier'][0]:.4f}, {ci['brier'][1]:.4f}]")
    print(f"    ECE:      [{ci['ece'][0]:.4f}, {ci['ece'][1]:.4f}]")

    # ── Save predictions ──
    out_df = df_labels.copy()
    out_df["pred_label"] = pred_labels
    out_df["confidence"] = p_selected
    out_df["is_correct"] = is_correct

    if num_classes == 2:
        # Only keep prob_1 for binary
        out_df["prob_1"] = probs[:, 1]
    else:
        for i in range(num_classes):
            col_name = f"prob_{inv_labels[i]}" if inv_labels and i in inv_labels else f"prob_{i}"
            out_df[col_name] = probs[:, i]

    out_df = out_df.reset_index()

    output_csv = args.output_csv or (args.data_dir / "predictions.csv")
    out_df.to_csv(output_csv, index=False)
    print(f"\nPredictions saved to: {output_csv}")

    print(
        f"\nSummary: ACC {acc_point:.4f} [{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}] | "
        f"Macro ACC {macro_acc:.4f} | "
        f"AUC {auc_point:.4f} [{ci['auc'][0]:.4f}, {ci['auc'][1]:.4f}] | "
        f"Brier {brier_point:.4f} [{ci['brier'][0]:.4f}, {ci['brier'][1]:.4f}] | "
        f"ECE {ece_point:.4f} [{ci['ece'][0]:.4f}, {ci['ece'][1]:.4f}]"
    )


if __name__ == "__main__":
    main()