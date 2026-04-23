#!/usr/bin/env python3
"""
calib_compare_overlay.py

Make a single-row figure:
  • If --layer_json is given: 1×3 panels
      [Layer metric vs. layer] | [Confidence histogram (overlay)] | [Calibration (overlay)]
  • Otherwise: 1×2 panels
      [Confidence histogram (overlay)] | [Calibration (overlay)]

Color scheme is configurable via CLI:
  - --ping_color:       used for the layer plot AND Ping in the other two
  - --generative_color: used for Generative in the other two

Usage:
  python calib_compare_overlay.py \
      --ping_csv  /path/to/ping/artifacts/<layer>/predictions.csv \
      --gen_csv   /path/to/mmlu_gemma2_test_top_token.csv \
      --out       mmlu_cc_overlay.png \
      --calib_bins 10 \
      --hist_bins  30 \
      --ping_name  "Ping (L41)" \
      --gen_name   "Gemma-2 2B-it" \
      [--hist_density] \
      [--layer_json /path/to/Layer_eval_*.json] \
      [--ping_color tab:green] \
      [--generative_color tab:blue] \
      [--dpi 300]

Notes:
  - ECE is the center-based variant from your original plotting code:
      sum_bin (weight_bin * |bin_acc - bin_center|)
  - For Ping CSV we prefer prob_A..prob_D (+ true_label/is_correct).
    If per-class probs are missing, we fall back to confidence+is_correct.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

LABEL_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

def _token_to_class_id(tok: str) -> float:
    """Map raw token to class index 0–3, else np.nan."""
    if not isinstance(tok, str):
        return np.nan
    letter = tok.lstrip("▁Ġ ").upper()
    return float(LABEL_MAP.get(letter, np.nan))

def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x.astype(float), 1e-12, 1 - 1e-12)

def compute_bin_accuracy(conf: np.ndarray, correct: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Bin by 'conf' using 'edges', then compute mean correctness per bin.
    Returns an array of length len(edges)-1 with NaN for empty bins.
    """
    idx = np.digitize(conf, edges, right=True) - 1
    out = np.full(len(edges) - 1, np.nan, dtype=float)
    for i in range(len(out)):
        m = (idx == i)
        if m.any():
            out[i] = correct[m].mean()
    return out

def compute_center_based_ece(conf: np.ndarray, correct: np.ndarray, edges: np.ndarray):
    """
    Your original ECE: weighted |bin_acc - bin_center|.
    Returns (ece, bin_acc, centers).
    """
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_acc = compute_bin_accuracy(conf, correct, edges)
    counts, _ = np.histogram(conf, edges)
    n = len(conf)
    ece = 0.0
    for i in range(len(centers)):
        if counts[i] > 0 and not np.isnan(bin_acc[i]):
            ece += (counts[i] / n) * abs(bin_acc[i] - centers[i])
    return float(ece), bin_acc, centers


# -----------------------------
# Load Ping predictions
# -----------------------------

def load_ping(ping_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (conf, correct) arrays for the Ping classifier.

    Accepts either:
      - per-class probabilities columns: prob_A..prob_D  (preferred)
        (plus either true_label or is_correct)
      - OR fallback columns: confidence + is_correct
      - OR pred_label + true_label + confidence
    """
    df = pd.read_csv(ping_csv)

    # Best case: per-class probs
    prob_cols = ["prob_A", "prob_B", "prob_C", "prob_D"]
    if all(c in df.columns for c in prob_cols):
        probs = df[prob_cols].to_numpy(dtype=float)
        pred = np.argmax(probs, axis=1)
        conf = _clip01(probs[np.arange(len(probs)), pred])

        if "true_label" in df.columns:
            true = df["true_label"].to_numpy()
            correct = (pred == true).astype(int)
        elif "is_correct" in df.columns:
            correct = df["is_correct"].to_numpy().astype(int)
        else:
            raise ValueError("Ping CSV with prob_* must include 'true_label' or 'is_correct'.")
        return conf, correct

    # Fallback: confidence + is_correct
    if "confidence" in df.columns and "is_correct" in df.columns:
        conf = _clip01(df["confidence"].to_numpy())
        correct = df["is_correct"].to_numpy().astype(int)
        return conf, correct

    # Fallback: pred_label + true_label + confidence
    if "pred_label" in df.columns and "true_label" in df.columns and "confidence" in df.columns:
        conf = _clip01(df["confidence"].to_numpy())
        correct = (df["pred_label"].to_numpy() == df["true_label"].to_numpy()).astype(int)
        return conf, correct

    raise ValueError(
        "Ping CSV: expected columns either ['prob_A'..'prob_D'] OR ('confidence','is_correct') "
        "OR ('pred_label','true_label','confidence')."
    )


# -----------------------------
# Load Generative results
# -----------------------------

def load_generative(gen_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (conf, correct) arrays for the Generative run,
    following your evaluation logic:
      - correctness from comparing mapped A–D of top_token to true label
      - confidence = top_token_prob if token is A–D, else 0
    """
    df = pd.read_csv(gen_csv)
    for c in ("correct", "top_token", "top_token_prob"):
        if c not in df.columns:
            raise ValueError(f"Generative CSV is missing required column: '{c}'")

    true = df["correct"].map(LABEL_MAP).to_numpy()
    pred_raw = df["top_token"].apply(_token_to_class_id).to_numpy()  # float with NaN possible
    pred = np.where(np.isnan(pred_raw), -1, pred_raw).astype(int)

    conf = df["top_token_prob"].to_numpy(dtype=float)
    conf = np.where(np.isnan(pred_raw), 0.0, conf)
    conf = _clip01(conf)

    correct = (pred == true).astype(int)
    return conf, correct


# -----------------------------
# Layer JSON loading & plotting
# -----------------------------

def _parse_layer_key(k) -> Optional[int]:
    """Try to coerce a JSON key to an int layer index, even if it's like 'L41'."""
    if isinstance(k, int):
        return k
    if isinstance(k, str):
        # prefer straight int string
        try:
            return int(k)
        except ValueError:
            m = re.search(r"\d+", k)
            if m:
                return int(m.group(0))
    return None

def load_layer_json(json_path: Path) -> Tuple[List[int], List[float], List[float], List[float], str, str]:
    """
    Returns (layers_sorted, mean_vals, min_vals, max_vals, metric_name, model_type)
    from a layer_eval-style JSON with {"results": {layer: {mean,min,max}}, "args": {...}}.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    if "results" not in data or "args" not in data:
        raise ValueError("Invalid JSON format. Expected 'results' and 'args' keys.")

    results = data["results"]
    layers: List[int] = []
    means:  List[float] = []
    mins:   List[float] = []
    maxs:   List[float] = []
    for k, v in results.items():
        li = _parse_layer_key(k)
        if li is None:
            continue
        means.append(float(v["mean"]))
        mins.append(float(v["min"]))
        maxs.append(float(v["max"]))
        layers.append(li)

    if not layers:
        raise ValueError("No numeric layer keys could be parsed from JSON 'results'.")

    # sort by layer
    order = np.argsort(layers)
    layers_sorted = list(np.array(layers)[order])
    mean_vals = list(np.array(means)[order])
    min_vals  = list(np.array(mins)[order])
    max_vals  = list(np.array(maxs)[order])

    args = data["args"]
    metric_name = str(args.get("metric", "metric"))
    model_type  = str(args.get("model_type", "model"))
    return layers_sorted, mean_vals, min_vals, max_vals, metric_name, model_type

def plot_layer_panel(
    ax: plt.Axes,
    layers_sorted: List[int],
    mean_vals: List[float],
    min_vals: List[float],
    max_vals: List[float],
    metric_name: str,
    legend_name: str,
    color: str = "tab:purple",
):
    # Uppercase first letter of the metric for y-label
    y_label = metric_name[:1].upper() + metric_name[1:]

    ax.plot(
        layers_sorted, mean_vals, marker="o", linewidth=2, markersize=6,
        color=color, label=legend_name
    )
    ax.fill_between(layers_sorted, min_vals, max_vals, alpha=0.25, color=color)

    # Labels/titles per request: remove x-axis label and title; keep y-label
    ax.set_xlabel("")         # intentionally blank
    ax.set_ylabel(y_label)
    # no title
    ax.grid(True, alpha=0.3)
    

    # reasonable integer ticks if many layers
    if len(layers_sorted) <= 50:
        step = max(1, len(layers_sorted)//10)
        ax.set_xticks(layers_sorted[::step])

    ax.legend()


# -----------------------------
# Figure (overlay; 1×2 or 1×3)
# -----------------------------

def make_figure(
    ping_conf: np.ndarray,
    ping_corr: np.ndarray,
    gen_conf:  np.ndarray,
    gen_corr:  np.ndarray,
    out_path:  str,
    calib_bins: int = 10,
    hist_bins:  int = 30,
    ping_name: str = "Ping",
    gen_name:  str = "Generative",
    hist_density: bool = False,
    layer_json: Optional[Path] = None,
    ping_color: str = "tab:green",
    generative_color: str = "tab:blue",
    dpi: int = 300,
):
    plt.rcParams.update({"font.size": 12})

    # Calibration prep
    edges = np.linspace(0.0, 1.0, calib_bins + 1)
    ece_g, binacc_g, centers = compute_center_based_ece(gen_conf,  gen_corr, edges)
    ece_p, binacc_p, _       = compute_center_based_ece(ping_conf, ping_corr, edges)

    include_layer = layer_json is not None

    # Figure layout
    if include_layer:
        figsize = (18, 3)
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=False)
        ax_layer, ax_hist, ax_cal = axes
    else:
        figsize = (8, 3)
        fig, (ax_hist, ax_cal) = plt.subplots(1, 2, figsize=figsize, sharex=False)
        ax_layer = None  # not used

    # -------- Optional: Layer panel --------
    if include_layer:
        Ls, means, mins, maxs, metric_name, model_type = load_layer_json(Path(layer_json))
        plot_layer_panel(
            ax_layer, Ls, means, mins, maxs,
            metric_name, gen_name,
            color=ping_color  # use Ping color for the first plot
        )

    # -------- Histogram (overlay) --------
    hist_kwargs = dict(bins=hist_bins, range=(0, 1), edgecolor="black", alpha=0.6, density=hist_density)
    n_g = len(gen_conf)
    n_p = len(ping_conf)

    ax_hist.hist(gen_conf, color=generative_color,  label=f"Generative", **hist_kwargs)
    ax_hist.hist(ping_conf, color=ping_color, label=f"{ping_name}", **hist_kwargs)

    # Remove title and x-axis label; keep y
    # ax_hist.set_title("")  # (no title)
    ax_hist.set_xlabel("")   # (no x label)
    ax_hist.set_ylabel("Density" if hist_density else "Count")
    ax_hist.grid(alpha=0.3, ls=":")
    # Histogram panel
    ax_hist.legend(fontsize=11, loc="upper left")

    # -------- Calibration (overlay) --------
    ax_cal.plot(centers, binacc_g, marker="o", lw=2, color=generative_color, label="Generative")
    ax_cal.plot(centers, binacc_p, marker="s", lw=2, color=ping_color, label=ping_name)
    ax_cal.plot([0, 1], [0, 1], ls="--", color="gray", label="Perfect")

    # Annotate ECE bottom right (relative coords: x=0.98, y=0.02)
    ax_cal.text(
        0.98, 0.02, f"ECE={ece_g:.3f}",
        transform=ax_cal.transAxes, ha="right", va="bottom",
        color=generative_color, fontsize=9
    )
    ax_cal.text(
        0.98, 0.10, f"ECE={ece_p:.3f}",
        transform=ax_cal.transAxes, ha="right", va="bottom",
        color=ping_color, fontsize=9
    )

    # Remove title and x-axis label; keep y
    # ax_cal.set_title("")   # (no title)
    ax_cal.set_xlabel("")    # (no x label)
    ax_cal.set_ylabel("Accuracy")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.grid(alpha=0.3, ls=":")

    # Calibration panel

    
    ax_cal.legend(fontsize=11, loc="upper left")


    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    print(f"Saved figure ➜ {out_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Overlay calibration+histogram plots for Ping vs Generative, with optional layer panel.")
    p.add_argument("--ping_csv", required=True, help="Path to Ping predictions.csv")
    p.add_argument("--gen_csv",  required=True, help="Path to Generative top-token CSV")
    p.add_argument("--out", default="mmlu_cc_overlay.png", help="Output PNG path")
    p.add_argument("--calib_bins", type=int, default=10, help="Number of bins for calibration/ECE")
    p.add_argument("--hist_bins",  type=int, default=30, help="Number of histogram bins")
    p.add_argument("--ping_name", default="Ping", help="Label for Ping model")
    p.add_argument("--gen_name",  default="Generative", help="Label for Generative model")
    p.add_argument("--hist_density", action="store_true",
                   help="Plot histogram as density instead of raw counts (useful if n differs).")
    # Layer JSON options
    p.add_argument("--layer_json", type=Path, default=None,
                   help="Optional: path to layer_eval JSON to render as the first panel.")
    # Colors
    p.add_argument("--ping_color", type=str, default="tab:green",
                   help="Color for Ping traces and the layer panel (default: tab:green).")
    p.add_argument("--generative_color", type=str, default="tab:gray",
                   help="Color for Generative traces (default: tab:gray).")
    p.add_argument("--dpi", type=int, default=300, help="Output figure DPI (default: 300).")

    args = p.parse_args()

    # Load predictions
    ping_conf, ping_corr = load_ping(args.ping_csv)
    gen_conf,  gen_corr  = load_generative(args.gen_csv)

    # Sanity checks
    for name, conf, corr in [("Ping", ping_conf, ping_corr), ("Generative", gen_conf, gen_corr)]:
        if len(conf) != len(corr):
            raise ValueError(f"{name}: conf and correctness lengths differ: {len(conf)} vs {len(corr)}")
        if not np.isfinite(conf).all():
            raise ValueError(f"{name}: found non-finite values in confidence.")
        if set(np.unique(corr)) - {0, 1}:
            raise ValueError(f"{name}: correctness must be 0/1.")

    make_figure(
        ping_conf, ping_corr,
        gen_conf,  gen_corr,
        out_path=args.out,
        calib_bins=args.calib_bins,
        hist_bins=args.hist_bins,
        ping_name=args.ping_name,
        gen_name=args.gen_name,
        hist_density=args.hist_density,
        layer_json=args.layer_json,
        ping_color=args.ping_color,
        generative_color=args.generative_color,
        dpi=args.dpi,
    )

if __name__ == "__main__":
    main()
