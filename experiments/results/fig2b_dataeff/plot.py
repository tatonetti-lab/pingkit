#!/usr/bin/env python
"""
plot_train_size_results.py
==========================

Plot training size sweep results with error bands, supporting single or multiple JSON files.

Defaults:
- Log scale with base 10 for x-axis
- Min/max error bands (use --show_std for ±1σ)

Key options:
- --metrics: Choose accuracy, brier, ece, or all
- --log_scale: Use log scale for x-axis (default: linear)
- --log_base: Set log base when using log scale (default: 10)
- --x_percent: Show x-axis as percentage of full training set
- --show_std: Show standard deviation bands instead of min/max
- --colors_file: Optional mapping of label->color to control line colors
- --save_colors: Save the actual colors used (label,color) to a text file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import (
    LogLocator,
    LogFormatter,
    MaxNLocator,
    AutoMinorLocator,
    FixedLocator,
    FuncFormatter,
    NullFormatter,
)


# ---------------------- CLI ---------------------- #
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training size impact results.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--json_file", type=Path, help="Single JSON file from train_size_sweep.py")
    src.add_argument("--dir", type=Path, help="Directory containing JSON files to plot.")
    
    p.add_argument(
        "--glob",
        type=str,
        default="train_size_*.json",
        help="Glob for JSON files when using --dir (default: train_size_*.json)",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd() / "plots",
        help="Output directory for PDF/PNG (default: ./plots)",
    )
    p.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output filename (without extension). If not set: "
             "'train_size_compare' for --dir; derived from JSON filename for --json_file.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=["accuracy", "brier", "ece", "all"],
        default=["accuracy"],
        help="Metrics to plot (default: accuracy)",
    )
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[9, 5],
        help="Figure size as width height (default: 9 5).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output (default: 300).",
    )
    p.add_argument(
        "--log_scale",
        action="store_true",
        help="Use log scale for x-axis (training sizes).",
    )
    p.add_argument(
        "--log_base",
        type=float,
        default=10.0,
        help="Logarithm base for x-axis when --log_scale is set (default: 10).",
    )
    p.add_argument(
        "--no_minor_ticks",
        action="store_true",
        help="Disable minor ticks/grid on the x-axis.",
    )
    p.add_argument(
        "--x_percent",
        action="store_true",
        help="Show x-axis as percentage of full training set.",
    )
    p.add_argument(
        "--show_std",
        action="store_true",
        help="Show standard deviation bands instead of min/max.",
    )
    p.add_argument(
        "--colors_file",
        type=Path,
        default=None,
        help="Optional text file mapping 'label,color' (CSV/TSV/space or 'label: color').",
    )
    p.add_argument(
        "--save_colors",
        nargs="?",
        const=True,
        default=False,
        help="Save the label->color mapping used. If a path is provided, save there; "
             "otherwise writes to <output_dir>/<output_name>_colors.txt",
    )
    p.add_argument(
        "--no_png",
        action="store_true",
        help="If set, do not also save a PNG (PDF is always saved).",
    )
    
    return p.parse_args(argv)


# ---------------------- IO ---------------------- #
def load_results(json_path: Path) -> Dict:
    if not json_path.exists():
        sys.exit(f"JSON file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    if "results" not in data:
        sys.exit(f"Invalid JSON format in {json_path}. Expected 'results' key.")
    return data


def find_jsons_in_dir(d: Path, pattern: str) -> List[Path]:
    files = sorted(d.glob(pattern))
    if not files:
        sys.exit(f"No files matched '{pattern}' in {d}")
    return files


def derive_output_name_from_file(f: Path) -> str:
    stem = f.stem
    if stem.startswith("train_size_"):
        return stem[11:]  # Remove 'train_size_' prefix
    return stem


# ---------------------- Helpers ---------------------- #
def extract_label_from_data(data: Dict, json_path: Path) -> str:
    """Extract a meaningful label from the data or filename."""
    if "args" in data:
        args = data["args"]
        if "train_dir" in args:
            # Try to extract meaningful component from train_dir
            path_parts = Path(args["train_dir"]).parts
            for part in reversed(path_parts):
                if "train" not in part.lower():
                    return part
    # Fallback to filename
    return derive_output_name_from_file(json_path)


def extract_series(data: Dict, metric: str) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """
    Extract training sizes and metric values from results.
    Returns: (sizes, means, mins, maxs, stds)
    """
    results = data["results"]
    
    sizes: List[int] = []
    means: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []
    stds: List[float] = []
    
    # Sort by actual size value
    for size_str in sorted(results.keys(), key=lambda k: int(k)):
        size = int(size_str)
        if metric in results[size_str]:
            rec = results[size_str][metric]
            sizes.append(size)
            means.append(rec["mean"])
            mins.append(rec["min"])
            maxs.append(rec["max"])
            stds.append(rec.get("std", 0.0))
    
    return sizes, means, mins, maxs, stds


def read_colors_file(path: Path) -> Dict[str, str]:
    """
    Reads lines like:
      label,color
      label\tcolor
      label color
      label: color
    Ignores blanks and lines starting with '#'.
    """
    color_mapping: Dict[str, str] = {}
    
    if not path.exists():
        sys.exit(f"Colors file not found: {path}")
    
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = []
            if "," in line:
                parts = [s.strip() for s in line.split(",")]
            elif "\t" in line:
                parts = [s.strip() for s in line.split("\t")]
            elif ":" in line:
                parts = [s.strip() for s in line.split(":")]
            else:
                parts = line.split()
            
            if len(parts) >= 2:
                label, color = parts[0], parts[1]
                color_mapping[label] = color
                
    return color_mapping


def save_colors_file(mapping: Dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# label,color\n")
        for label, color in mapping.items():
            f.write(f"{label},{color}\n")
    print(f"Saved colors mapping to {path}")


def determine_metric_label(metric: str) -> str:
    labels = {
        "accuracy": "Accuracy",
        "brier": "Brier Score",
        "ece": "Expected Calibration Error (ECE)"
    }
    return labels.get(metric, metric.capitalize())


# ---------------------- Plotting ---------------------- #
def plot_multiple(
    json_files: List[Path],
    metrics: List[str],
    figsize: Tuple[float, float],
    dpi: int,
    log_scale: bool,
    log_base: float,
    x_percent: bool,
    no_minor_ticks: bool,
    show_std: bool,
    input_colors: Optional[Dict[str, str]] = None,
) -> Tuple[plt.Figure, Dict[str, str], List[str]]:
    """Plot metrics for all JSON files and return the figure, color mapping, and labels."""
    
    n_metrics = len(metrics)
    
    # Create figure with subplots if needed
    if n_metrics == 1:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True, dpi=dpi)
        axes = [ax]
    else:
        n_cols = min(3, n_metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(figsize[0] * n_cols / 2, figsize[1] * n_rows / 2),
            constrained_layout=True,
            dpi=dpi,
        )
        axes = axes.flatten() if n_metrics > 1 else [axes]
    
    # Colors setup
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = list(mcolors.TABLEAU_COLORS.values())
    used_color_map: Dict[str, str] = {}
    used_hexes = set()
    
    all_labels: List[str] = []
    
    # Plot each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Track data for axis limits
        all_x: List[float] = []
        max_train_size = 0
        
        # Plot each series
        for file_idx, json_path in enumerate(json_files):
            try:
                data = load_results(json_path)
            except SystemExit as e:
                print(f"[WARN] Skipping {json_path}: {e}")
                continue
            
            label = extract_label_from_data(data, json_path)
            if metric_idx == 0:  # Only collect labels once
                all_labels.append(label)
            
            sizes, means, mins, maxs, stds = extract_series(data, metric)
            if not sizes:
                print(f"[WARN] No data for metric '{metric}' in {json_path}")
                continue
            
            # Track max train size for percentage mode
            if "dataset_info" in data:
                max_train_size = max(
                    max_train_size,
                    data["dataset_info"].get("full_train_size", max(sizes))
                )
            
            # Prepare x values
            if x_percent and max_train_size > 0:
                x_vals = [(s / max_train_size) * 100.0 for s in sizes]
            else:
                x_vals = sizes
            
            all_x.extend(x_vals)
            
            # Choose color
            color = None
            if input_colors and label in input_colors:
                color = input_colors[label]
            elif label in used_color_map:
                color = used_color_map[label]
            else:
                for c in color_cycle:
                    c_hex = mcolors.to_hex(c)
                    if c_hex not in used_hexes:
                        color = c
                        break
                if color is None:
                    color = color_cycle[file_idx % len(color_cycle)]
            
            # Error bands
            if show_std:
                lower = [m - s for m, s in zip(means, stds)]
                upper = [m + s for m, s in zip(means, stds)]
                ax.fill_between(x_vals, lower, upper, alpha=0.20, color=color, linewidth=0)
            else:
                ax.fill_between(x_vals, mins, maxs, alpha=0.15, color=color, linewidth=0)
            
            # Mean line
            line, = ax.plot(
                x_vals, means, marker="o", linewidth=1.8, markersize=4, label=label, color=color
            )
            
            actual_color = mcolors.to_hex(line.get_color())
            used_color_map[label] = actual_color
            used_hexes.add(actual_color)
        
        # Configure axes
        if all_x:
            configure_x_axis(ax, all_x, log_scale, log_base, x_percent, not no_minor_ticks)
            
            # Add special minor ticks for log scale
            if log_scale and not x_percent:
                add_special_log_ticks(ax, [50.0,250.0,500.0,5000.0])
            
            # Add ONLY the requested vertical grid lines at x = 100, 1000, 5000
            draw_vertical_guides_at_x(
                ax=ax,
                positions=[50.0,100.0, 250.0,500.0, 1000.0, 5000.0],
                x_percent=x_percent,
                max_train_size=max_train_size,
            )
        
        # Y-axis setup
        ax.set_ylabel(determine_metric_label(metric), fontsize=12)
        if metric in ["brier", "ece"]:
            ax.set_ylim(bottom=0)
        
        # Disable default grids
        ax.grid(False)
        
        # Legend
        if len(json_files) > 1:
            n_cols = 1 if len(json_files) <= 12 else 2
            ax.legend(
                title="Model" if metric_idx == 0 else None,
                loc="best",
                frameon=True,
                borderaxespad=0.5,
                ncol=n_cols,
                fontsize=11
            )
        
        # Subplot title for multi-metric plots
        if n_metrics > 1:
            ax.set_title(determine_metric_label(metric), fontsize=12)
    
    # Hide unused subplots
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)
    
    # Figure title
    if n_metrics == 1:
        fig.suptitle(f"Training Size Impact on {determine_metric_label(metrics[0])}", fontsize=14)
    else:
        fig.suptitle("Training Size Impact on Model Performance", fontsize=14)
    
    return fig, used_color_map, all_labels


def configure_x_axis(
    ax: plt.Axes,
    x_vals: List[float],
    log_scale: bool,
    log_base: float,
    x_percent: bool,
    show_minor: bool
) -> None:
    """Configure x-axis ticks, formatters, limits, and grid."""
    
    # Set label
    if x_percent:
        ax.set_xlabel("Training Set Size (% of full dataset)", fontsize=12)
    else:
        ax.set_xlabel("Training Set Size (number of examples)", fontsize=12)
    
    xmin, xmax = float(np.min(x_vals)), float(np.max(x_vals))
    
    if log_scale and not x_percent:
        if xmin <= 0:
            raise ValueError("Log scale requested but x contains non-positive values.")
        ax.set_xscale("log", base=log_base)
        
        # Major ticks
        major = LogLocator(base=log_base, numticks=12)
        ax.xaxis.set_major_locator(major)
        ax.xaxis.set_major_formatter(LogFormatter(base=log_base, labelOnlyBase=False))
        
        # Minor ticks
        if show_minor:
            upper = int(min(9, max(2, int(log_base) - 1)))
            subs = (np.arange(2, upper + 1) / log_base).tolist()
            ax.xaxis.set_minor_locator(LogLocator(base=log_base, subs=subs, numticks=100))
            ax.tick_params(axis="x", which="minor", length=3)
        
        # Grid
        ax.grid(False)
        if show_minor:
            ax.grid(False)
        
        # Limits
        ax.set_xlim(xmin / (log_base ** 0.05), xmax * (log_base ** 0.05))
    
    elif x_percent:
        # Percentage mode
        ax.set_xlim(0, 100)
        ax.set_xticks(list(range(0, 101, 10)))
        ax.set_xticklabels([f"{v}%" for v in range(0, 101, 10)])
        ax.grid(False)
    
    else:
        # Linear mode
        ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 2, 5, 10]))
        if show_minor:
            ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
            ax.grid(False)
        ax.grid(False)
        pad = 0.02 * (xmax - xmin if xmax > xmin else 1.0)
        ax.set_xlim(xmin - pad, xmax + pad)


def add_special_log_ticks(ax: plt.Axes, special_vals: List[float]) -> None:
    """Add specific minor tick values with labels on log axis."""
    lo, hi = ax.get_xlim()
    if lo > 0 and hi > 0:
        # Keep only values in view
        vals = [float(v) for v in special_vals if (v > 0 and lo <= v <= hi)]
        
        # Remove any that coincide with major ticks
        majors = np.asarray(ax.xaxis.get_majorticklocs(), dtype=float)
        vals = [v for v in vals if not np.any(np.isclose(majors, v, rtol=1e-8, atol=1e-10))]
        
        # Apply as minor ticks
        ax.xaxis.set_minor_locator(FixedLocator(vals))
        
        if vals:
            vals_arr = np.asarray(vals, dtype=float)
            ax.xaxis.set_minor_formatter(
                FuncFormatter(lambda v, pos, _vals=vals_arr: f"{int(round(v))}" 
                            if np.any(np.isclose(v, _vals, rtol=1e-8, atol=1e-10)) else "")
            )
            ax.tick_params(axis="x", which="minor", length=3,
                        labelsize=plt.rcParams.get("xtick.labelsize", 10))
        else:
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.xaxis.set_minor_formatter(NullFormatter())


def draw_vertical_guides_at_x(
    ax: plt.Axes,
    positions: List[float],
    x_percent: bool,
    max_train_size: int,
) -> None:
    """
    Draw vertical guide lines (grid-like) ONLY at the specified x positions.
    If x_percent is True and max_train_size > 0, convert absolute sizes to percentages.
    """
    # Determine where to draw (convert to % if needed)
    if x_percent:
        if max_train_size and max_train_size > 0:
            xs = [(p / max_train_size) * 100.0 for p in positions]
        else:
            # Cannot convert without a known full_train_size
            xs = []
    else:
        xs = positions

    if not xs:
        return

    lo, hi = ax.get_xlim()
    grid_color = plt.rcParams.get("grid.color", "0.75")
    grid_alpha = 0.35
    grid_linestyle = "--"
    grid_linewidth = 0.8

    # Only draw lines that fall within the current view
    for x in xs:
        if lo <= x <= hi:
            ax.axvline(x=x, linestyle=grid_linestyle, linewidth=grid_linewidth, color=grid_color, alpha=grid_alpha)


# ---------------------- Main ---------------------- #
def main(argv: List[str] | None = None) -> None:
    plt.rcParams.update({"font.size": 12})
    args = parse_args(argv)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine input files
    if args.json_file:
        json_files = [args.json_file]
        output_name = args.output_name or derive_output_name_from_file(args.json_file)
    else:
        json_files = find_jsons_in_dir(args.dir, args.glob)
        output_name = args.output_name or "train_size_compare"
    
    out_pdf = args.output_dir / f"{output_name}.pdf"
    out_png = args.output_dir / f"{output_name}.png"
    
    # Process metrics
    metrics = ["accuracy", "brier", "ece"] if "all" in args.metrics else args.metrics
    
    # Load colors if provided
    input_colors = read_colors_file(args.colors_file) if args.colors_file else None
    
    # Print info
    print(f"Found {len(json_files)} file(s).")
    for f in json_files:
        print(f"  - {f.name}")
    
    # Create plot
    fig, used_colors, labels = plot_multiple(
        json_files,
        metrics,
        tuple(args.figsize),
        args.dpi,
        args.log_scale,
        args.log_base,
        args.x_percent,
        args.no_minor_ticks,
        args.show_std,
        input_colors
    )
    
    # Save outputs
    print(f"Saving to {out_pdf}")
    fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight")
    if not args.no_png:
        fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    
    # Save colors if requested
    if args.save_colors:
        if isinstance(args.save_colors, str):
            colors_out = Path(args.save_colors)
        else:
            colors_out = args.output_dir / f"{output_name}_colors.txt"
        # Keep order consistent with labels
        ordered: Dict[str, str] = {}
        for label in labels:
            if label in used_colors:
                ordered[label] = used_colors[label]
        for k, v in used_colors.items():
            if k not in ordered:
                ordered[k] = v
        save_colors_file(ordered, colors_out)
    
    # Print summary
    print("\nSummary:")
    print(f"  Mode: {'single file' if args.json_file else 'directory'}")
    print(f"  Output name: {output_name}")
    print(f"  X-axis: {'log scale (base ' + str(args.log_base) + ')' if args.log_scale else 'linear scale'}")
    if args.x_percent:
        print(f"  X-axis display: % of full dataset")
    else:
        print(f"  X-axis display: number of examples")
    print(f"  Metrics plotted: {', '.join(metrics)}")
    if args.show_std:
        print("  Error bands: standard deviation (±1σ)")
    else:
        print("  Error bands: min/max range")
    if args.colors_file:
        print(f"  Colors input: {args.colors_file}")
    if args.save_colors:
        print("  Colors saved.")


if __name__ == "__main__":
    main()
