#!/usr/bin/env python
"""
plot_layer_results.py
====================

Reads JSON results from layer_eval.py and generates a PDF plot showing
the metric performance across transformer layers.

-----------------------------------------------------------------------
Example
-------
$ python plot_layer_results.py \
      --json_file Layer_eval_mmlu_g_20240101_120000.json \
      --output_dir plots/
-----------------------------------------------------------------------

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(description="Plot layer evaluation results from JSON.")

    p.add_argument(
        "--json_file",
        required=True,
        type=Path,
        help="JSON file containing layer evaluation results.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for PDF. If not specified, saves next to JSON file.",
    )
    p.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output filename (without extension). If not specified, derives from JSON filename.",
    )
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[8, 4],
        help="Figure size as width height (default: 8 4).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF output (default: 300).",
    )

    return p.parse_args(argv)


def load_results(json_path: Path) -> tuple[dict, dict]:
    """Load results and args from JSON file."""
    if not json_path.exists():
        sys.exit(f"JSON file not found: {json_path}")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if "results" not in data or "args" not in data:
        sys.exit(f"Invalid JSON format. Expected 'results' and 'args' keys.")
    
    return data["results"], data["args"]


def create_plot(results: dict, args: dict, figsize: tuple[float, float]) -> plt.Figure:
    """Create the layer evaluation plot."""
    # Extract data
    layers_sorted = sorted([int(k) for k in results.keys()])
    mean_vals = [results[str(l)]["mean"] for l in layers_sorted]
    min_vals = [results[str(l)]["min"] for l in layers_sorted]
    max_vals = [results[str(l)]["max"] for l in layers_sorted]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(layers_sorted, mean_vals, marker="o", linewidth=2, markersize=6)
    ax.fill_between(layers_sorted, min_vals, max_vals, alpha=0.3)
    
    # Styling
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel(args.get("metric", "metric"))
    ax.set_title(f"{args.get('metric', 'Metric')} vs. layer ({args.get('model_type', 'model')})")
    ax.grid(True, alpha=0.3)
    
    # Set integer ticks on x-axis if reasonable number of layers
    if len(layers_sorted) <= 50:
        ax.set_xticks(layers_sorted[::max(1, len(layers_sorted)//10)])
    
    plt.tight_layout()
    return fig


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    
    # Load results
    print(f"Loading results from {args.json_file}")
    results, eval_args = load_results(args.json_file)
    
    # Determine output path
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = args.json_file.parent
    
    if args.output_name:
        output_name = args.output_name
    else:
        # Derive from JSON filename, removing timestamp if present
        stem = args.json_file.stem
        if stem.startswith("Layer_eval_"):
            # Remove "Layer_eval_" prefix and timestamp suffix if present
            parts = stem.split("_")
            if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                # Remove timestamp (last two parts: YYYYMMDD_HHMMSS)
                output_name = "_".join(parts[2:-2]) if len(parts) > 4 else parts[2]
            else:
                output_name = "_".join(parts[2:])
        else:
            output_name = stem
    
    output_path = output_dir / f"{output_name}.pdf"
    
    # Create and save plot
    print(f"Creating plot...")
    fig = create_plot(results, eval_args, tuple(args.figsize))
    
    print(f"Saving plot to {output_path}")
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    
    # Print summary
    num_layers = len(results)
    metric = eval_args.get("metric", "metric")
    best_layer = max(results.keys(), key=lambda k: results[k]["mean"])
    best_score = results[best_layer]["mean"]
    
    print(f"\nSummary:")
    print(f"  Evaluated {num_layers} layers")
    print(f"  Best {metric}: {best_score:.4f} at layer {best_layer}")
    print(f"  Plot saved: {output_path}")


if __name__ == "__main__":
    main()