#!/usr/bin/env python
"""
plot.py — Plot per-part layer evaluation results from a single JSON.

Reads the output of layer_eval.py (which stores results keyed by part name)
and plots each part as its own series with min/max shaded bands.

Example:
    python plot.py --json_file my_results.json --output plots/my_plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot per-part layer eval results.")
    p.add_argument("--json_file", required=True, type=Path, help="JSON from layer_eval.py.")
    p.add_argument("--output", type=str, default=None,
                   help="Output path without extension (saves .pdf and .png). "
                        "Defaults to same directory/stem as input.")
    p.add_argument("--figsize", nargs=2, type=float, default=[9, 5],
                   help="Figure size: width height.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no_png", action="store_true", help="Skip PNG, only save PDF.")
    return p.parse_args(argv)


def load_json(path: Path) -> Tuple[dict, dict]:
    if not path.exists():
        sys.exit(f"File not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if "results" not in data or "args" not in data:
        sys.exit(f"Invalid JSON format in {path}.")
    return data["results"], data["args"]


def extract_series(part_results: dict) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Parse a part's {layer: {mean, min, max}} dict into sorted lists."""
    layers, means, mins, maxs = [], [], [], []
    for k, v in part_results.items():
        try:
            li = int(k)
        except (TypeError, ValueError):
            continue
        layers.append(li)
        means.append(v["mean"])
        mins.append(v.get("min", v["mean"]))
        maxs.append(v.get("max", v["mean"]))

    order = sorted(range(len(layers)), key=lambda i: layers[i])
    return (
        [layers[i] for i in order],
        [means[i] for i in order],
        [mins[i] for i in order],
        [maxs[i] for i in order],
    )


PART_COLORS: Dict[str, str] = {
    "rs": "#1f77b4",
    "attn": "#e377c2",
    "mlp": "#2ca02c",
}


def plot(
    results: dict,
    args_meta: dict,
    figsize: Tuple[float, float],
    dpi: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    color_cycle = list(mcolors.TABLEAU_COLORS.values())
    color_idx = 0

    for part_name, part_data in results.items():
        layers, means, mins, maxs = extract_series(part_data)
        if not layers:
            print(f"[WARN] No layers for part '{part_name}', skipping.")
            continue

        color = PART_COLORS.get(part_name)
        if color is None:
            color = color_cycle[color_idx % len(color_cycle)]
            color_idx += 1

        ax.fill_between(layers, mins, maxs, alpha=0.15, color=color, linewidth=0)
        ax.plot(layers, means, marker="o", linewidth=1.8, markersize=4,
                color=color, label=part_name)

    metric = args_meta.get("metric", "accuracy")
    ax.set_xlabel("Transformer layer", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"{metric.capitalize()} over Layers (by part)")
    ax.grid(False)

    # x-axis ticks
    all_layers = []
    for part_data in results.values():
        for k in part_data:
            try:
                all_layers.append(int(k))
            except (TypeError, ValueError):
                pass
    if all_layers:
        max_layer = max(all_layers)
        step = 5 if max_layer >= 25 else 2 if max_layer >= 10 else 1
        ax.set_xlim(0, max_layer)
        ax.set_xticks(range(0, max_layer + 1, step))

    ax.legend(title="Part", loc="lower right", frameon=True, fontsize=11)
    plt.tight_layout()
    return fig


def main(argv: List[str] | None = None) -> None:
    plt.rcParams.update({"font.size": 12})
    args = parse_args(argv)

    results, args_meta = load_json(args.json_file)

    if args.output:
        out_base = Path(args.output)
    else:
        out_base = args.json_file.with_suffix("")

    out_base.parent.mkdir(parents=True, exist_ok=True)

    fig = plot(results, args_meta, tuple(args.figsize), args.dpi)

    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(pdf_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {pdf_path}")

    if not args.no_png:
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
