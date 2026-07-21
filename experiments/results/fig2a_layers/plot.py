#!/usr/bin/env python
"""
plot.py
====================

Plot single or multiple layer-eval JSON results with optional generative accuracy baselines.

Defaults:
- Uses **all layers** by default (no clipping).

Key options:
- --layer_cutoff: optionally clip each series to a layer threshold.
    * Absolute layers: e.g., 50
    * Percentage of model depth: e.g., 80%
- --x_percent: normalize x-axis by percent of **each model's plotted depth** (0..100).
    * In percent mode, **every series extends to 100%** (flat to the border using its last y).
    * In raw-layer mode, only the deepest series pads to the right border; others stop naturally.
- --colors_file: optional mapping of label->color[,generative_accuracy] to control line colors and generative baselines.
- --show_generative: if set, plot dotted horizontal lines for generative accuracy from colors file.
- --save_colors: save the actual colors used (label,color[,generative]) to a text file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------- CLI ---------------------- #
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot layer evaluation results from JSON.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--json_file", type=Path, help="Single JSON file to plot.")
    src.add_argument("--dir", type=Path, help="Directory containing JSON files to plot.")

    p.add_argument(
        "--glob",
        type=str,
        default="Layer_eval_*.json",
        help="Glob for JSON files when using --dir (default: Layer_eval_*.json)",
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
             "'layers_compare' for --dir; derived from JSON filename for --json_file.",
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
        "--x_percent",
        action="store_true",
        help="If set, x-axis is 0..100 representing each model's plotted depth "
             "(i.e., up to cutoff if provided, else full model). In percent mode, "
             "every series extends to 100%.",
    )
    p.add_argument(
        "--layer_cutoff",
        type=str,
        default=None,
        help="Optional cutoff per model. Examples: '50' (first 50 layers), '10%%' (first 10%% of layers). "
             "Default: use all layers.",
    )
    p.add_argument(
        "--colors_file",
        type=Path,
        default=None,
        help="Optional text file mapping 'label,color[,generative_accuracy]' (CSV/TSV/space or 'label: color').",
    )
    p.add_argument(
        "--show_generative",
        action="store_true",
        help="If set, plot dotted horizontal lines for generative accuracy values from colors file.",
    )
    p.add_argument(
        "--save_colors",
        nargs="?",
        const=True,
        default=False,
        help="Save the label->color[->generative] mapping used. If a path is provided, save there; "
             "otherwise writes to <output_dir>/<output_name>_colors.txt",
    )
    p.add_argument(
        "--no_png",
        action="store_true",
        help="If set, do not also save a PNG (PDF is always saved).",
    )

    return p.parse_args(argv)


# ---------------------- IO ---------------------- #
def load_results(json_path: Path) -> Tuple[dict, dict]:
    if not json_path.exists():
        sys.exit(f"JSON file not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    if "results" not in data or "args" not in data:
        sys.exit(f"Invalid JSON format in {json_path}. Expected 'results' and 'args' keys.")
    return data["results"], data["args"]


def find_jsons_in_dir(d: Path, pattern: str) -> List[Path]:
    files = sorted(d.glob(pattern))
    if not files:
        sys.exit(f"No files matched '{pattern}' in {d}")
    return files


def derive_output_name_from_file(f: Path) -> str:
    stem = f.stem
    if stem.startswith("Layer_eval_"):
        parts = stem.split("_")
        # Try to strip timestamp suffix with two numeric parts (YYYYMMDD_HHMMSS)
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            return "_".join(parts[2:-2]) if len(parts) > 4 else parts[2]
        return "_".join(parts[2:])
    return stem


# ---------------------- Helpers ---------------------- #
def extract_model_from_embedding_dir(val: str) -> str:
    """
    Examples:
      'MMLU_gemma-2-9b-it_train' -> 'gemma-2-9b-it'
      'MMLU_llama-3-70b_instruct_val' -> 'llama-3-70b_instruct' (suffix kept if not a split)
    Strategy:
      - Split by '_' and drop the first token (dataset prefix).
      - Drop trailing tokens if they are common split names.
      - Join remaining with '_' if there are multiples.
    """
    if not isinstance(val, str):
        return "unknown"
    parts = val.split("_")
    middle = parts[1:] if len(parts) > 1 else parts
    suffixes = {"train", "test", "val", "dev", "valid", "validation"}
    if middle and middle[-1].lower() in suffixes:
        middle = middle[:-1]
    if not middle:
        return val
    return "_".join(middle)


def parse_layer_cutoff(spec: Optional[str]) -> Optional[Union[int, float]]:
    """
    Parses --layer_cutoff.
      - None: no cutoff (use all layers)
      - 'N'   -> returns int N (absolute layers)
      - 'P%' -> returns float P (percentage of depth)
    """
    if spec is None:
        return None
    s = spec.strip()
    if s.endswith("%"):
        try:
            p = float(s[:-1])
        except ValueError:
            sys.exit(f"Invalid percentage for --layer_cutoff: {spec}")
        # clamp
        if p < 0:
            p = 0.0
        if p > 100:
            p = 100.0
        return p
    else:
        try:
            n = int(s)
        except ValueError:
            sys.exit(f"Invalid integer for --layer_cutoff: {spec}")
        if n < 0:
            n = 0
        return n


def cutoff_to_count(total_layers: int, cutoff: Optional[Union[int, float]]) -> int:
    """
    Converts cutoff spec to a concrete layer-count (exclusive upper bound).
    - None => total_layers (i.e., all layers)
    - int N => min(N, total_layers)
    - float P (percent) => ceil(P% of total_layers)
    """
    if cutoff is None:
        return total_layers
    if isinstance(cutoff, int):
        return min(cutoff, total_layers)
    import math
    return min(total_layers, max(0, math.ceil((cutoff / 100.0) * total_layers)))


def series_from_results(
    results: dict,
    cutoff_spec: Optional[Union[int, float]],
) -> Tuple[List[int], List[float], List[float], List[float], int, int]:
    """
    Returns (layers_sorted, mean_values_sorted, min_values_sorted, max_values_sorted,
             total_layers, effective_count).

    - Keeps only integer-like layer keys < effective_count.
    - total_layers is estimated as max(layer_index)+1 across ALL integer-like keys.
    """
    # Parse all int-like keys
    int_keys = []
    for k in results.keys():
        try:
            int_keys.append(int(k))
        except (TypeError, ValueError):
            pass
    if not int_keys:
        return [], [], [], [], 0, 0

    total_layers = max(int_keys) + 1
    effective_count = cutoff_to_count(total_layers, cutoff_spec)

    # Clip and collect stats
    layers: List[int] = []
    means: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []
    for k in results.keys():
        try:
            li = int(k)
        except (TypeError, ValueError):
            continue
        if li < effective_count:
            rec = results[str(li)]
            m = rec.get("mean")
            # Fall back to mean if min/max absent
            mn = rec.get("min", m)
            mx = rec.get("max", m)
            layers.append(li)
            means.append(m)
            mins.append(mn)
            maxs.append(mx)

    # Sort by layer
    order = sorted(range(len(layers)), key=lambda i: layers[i])
    layers_sorted = [layers[i] for i in order]
    means_sorted = [means[i] for i in order]
    mins_sorted = [mins[i] for i in order]
    maxs_sorted = [maxs[i] for i in order]
    return layers_sorted, means_sorted, mins_sorted, maxs_sorted, total_layers, effective_count



def make_x_values_percent_per_model(layers: List[int], effective_count: int) -> List[float]:
    """
    Map layer index -> 0..100 by (layer / effective_count) * 100.
    Note: uses 0-based indices; last real point will be < 100 when layers are 0..effective_count-1.
    """
    denom = max(1, effective_count)
    return [(l / denom) * 100.0 for l in layers]


def read_colors_file(path: Path) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Reads lines like:
      label,color[,generative_accuracy]
      label\tcolor[\tgenerative_accuracy]
      label color [generative_accuracy]
      label: color[:generative_accuracy]
    Ignores blanks and lines starting with '#'.
    
    Returns: (color_mapping, generative_mapping)
    """
    color_mapping: Dict[str, str] = {}
    generative_mapping: Dict[str, float] = {}
    
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
            
            if len(parts) < 2:
                continue
                
            label, color = parts[0], parts[1]
            color_mapping[label] = color
            
            # Check for optional third column (generative accuracy)
            if len(parts) >= 3:
                try:
                    gen_acc = float(parts[2])
                    generative_mapping[label] = gen_acc
                except ValueError:
                    pass  # Ignore invalid generative accuracy values
                    
    return color_mapping, generative_mapping


def save_colors_file(mapping: Dict[str, str], generative_mapping: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# label,color,generative_accuracy\n")
        for label, color in mapping.items():
            if label in generative_mapping:
                f.write(f"{label},{color},{generative_mapping[label]:.4f}\n")
            else:
                f.write(f"{label},{color}\n")
    print(f"Saved colors mapping to {path}")


def label_with_depth(label: str, total_layers: int) -> str:
    if total_layers > 0:
        return f"{label} ({total_layers}L)"
    return label


def determine_metric_label(metrics: List[str]) -> str:
    uniq = {m for m in metrics if isinstance(m, str)}
    if len(uniq) == 1:
        return uniq.pop()
    return "metric"


# ---------------------- Plotting ---------------------- #
def plot_multiple(
    json_files: List[Path],
    figsize: Tuple[float, float],
    dpi: int,
    x_percent: bool,
    cutoff_spec: Optional[Union[int, float]],
    input_colors: Optional[Dict[str, str]] = None,
    generative_accs: Optional[Dict[str, float]] = None,
    show_generative: bool = False,
) -> Tuple[plt.Figure, Dict[str, str], Dict[str, float], List[str]]:
    # First pass: load/prepare series
    series = []
    metric_names: List[str] = []
    for fp in json_files:
        try:
            results, args = load_results(fp)
        except SystemExit as e:
            print(f"[WARN] Skipping {fp}: {e}")
            continue

        layers, means, mins, maxs, total_layers, effective_count = series_from_results(results, cutoff_spec)
        if not layers:
            print(f"[WARN] No layers found under cutoff in {fp}; skipping.")
            continue
        base_label = extract_model_from_embedding_dir(args.get("embedding_dir", "unknown"))
        metric_names.append(args.get("metric", "metric"))
        series.append({
            "file": fp,
            "base_label": base_label,
            "label": label_with_depth(base_label, total_layers),
            "layers": layers,
            "means": means,
            "mins": mins,
            "maxs": maxs,
            "total_layers": total_layers,
            "effective_count": effective_count,
        })

    if not series:
        sys.exit("No valid series to plot.")

    # Establish global info (raw-layer mode uses this); percent mode doesn't need it for scaling
    global_count = max(s["effective_count"] for s in series)  # max plotted depth across series
    pad_leader_done = False                                   # used only in raw-layer mode
    right_boundary = global_count                             # used only in raw-layer mode

    fig, ax = plt.subplots(figsize=figsize)

    # Colors
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = list(mcolors.TABLEAU_COLORS.values())
    used_color_map: Dict[str, str] = {}   # base_label -> hex color
    used_generative_map: Dict[str, float] = {}  # base_label -> generative accuracy
    used_hexes = set()                    # to avoid reusing same hex

    legend_labels: List[str] = []
    has_generative = False  # Track if we've plotted any generative lines

    # Plot each series
    for idx, s in enumerate(series):
        base_label = s["base_label"]
        label = s["label"]
        layers = s["layers"]
        means = s["means"][:]  # copy so we can append for padding without mutating original
        mins = s["mins"][:]
        maxs = s["maxs"][:]
        effective_count = s["effective_count"]

        if x_percent:
            # Percent of *this model's* plotted depth; pad EVERY series to 100%
            x_vals = make_x_values_percent_per_model(layers, effective_count)
            if x_vals and (x_vals[-1] < 100.0):
                x_vals = x_vals + [100.0]
                means = means + [means[-1]]
                mins = mins + [mins[-1]]
                maxs = maxs + [maxs[-1]]
        else:
            # Raw-layer axis; only the deepest series pads to global right boundary
            x_vals = layers[:]
            if (not pad_leader_done) and (effective_count == right_boundary) and x_vals and (x_vals[-1] < right_boundary):
                x_vals = x_vals + [right_boundary]
                means = means + [means[-1]]
                mins = mins + [mins[-1]]
                maxs = maxs + [maxs[-1]]
                pad_leader_done = True

        # Choose color: input mapping » existing used mapping » next in cycle
        color = None
        if input_colors and base_label in input_colors:
            color = input_colors[base_label]
        elif base_label in used_color_map:
            color = used_color_map[base_label]
        else:
            if color_cycle:
                for c in color_cycle:
                    c_hex = mcolors.to_hex(c)
                    if c_hex not in used_hexes:
                        color = c
                        break
            if color is None:
                color = color_cycle[idx % len(color_cycle)] if color_cycle else "C0"

        # Shaded error band first so the line draws on top
        ax.fill_between(x_vals, mins, maxs, alpha=0.15, color=color, linewidth=0)

        line, = ax.plot(x_vals, means, marker="o", linewidth=1.8, markersize=4, label=base_label, color=color)

        actual_color = mcolors.to_hex(line.get_color())
        used_color_map[base_label] = actual_color
        used_hexes.add(actual_color)
        legend_labels.append(base_label)
        
        # Plot generative accuracy line if available and enabled
        if show_generative and generative_accs and base_label in generative_accs:
            gen_acc = generative_accs[base_label]
            used_generative_map[base_label] = gen_acc
            
            # Get x-axis limits for the dotted line
            if x_percent:
                x_start, x_end = 0, 100
            else:
                x_start, x_end = 0, right_boundary
            
            # Plot dotted line in the same color as the model
            ax.axhline(y=gen_acc, xmin=0, xmax=1, color=actual_color, linestyle=':', linewidth=1.5, alpha=0.7)
            has_generative = True

    # Axes, ticks, title
    ax.set_xlabel("Layer (% of plotted depth)" if x_percent else "Transformer layer")
    y_label = "Accuracy"
    ax.set_ylabel(y_label, fontsize=12)

    # Title suffix
    if cutoff_spec is None:
        cutoff_spec_str = "all layers"
    elif isinstance(cutoff_spec, int):
        cutoff_spec_str = f"first {cutoff_spec} layers"
    else:
        cutoff_spec_str = f"first {cutoff_spec:.0f}% of layers"
    suffix = f"({cutoff_spec_str}; x = % depth)" if x_percent else f"({cutoff_spec_str})"
    ax.set_title(f"Accuracy over Layers")

    ax.grid(False)

    if x_percent:
        ax.set_xlim(0, 100)
        ax.set_xticks(list(range(0, 101, 10)))
        ax.set_xticklabels([f"{v}%" for v in range(0, 101, 10)])
        ax.tick_params(axis="x", labelsize=12)
    else:
        ax.set_xlim(0, right_boundary)
        rb = int(right_boundary)
        step = 5 if rb >= 25 else 2 if rb >= 10 else 1
        ax.set_xticks(list(range(0, rb + 1, step)))
        ax.tick_params(axis="x", labelsize=11)

    # Legend
    n_cols = 1 if len(series) <= 12 else 2
    
    # Create custom legend handles if we have generative lines
    if has_generative:
        from matplotlib.lines import Line2D
        handles, labels = ax.get_legend_handles_labels()
        
        # Add a black dotted line for "Generative Performance"
        generative_line = Line2D([0], [0], color='black', linestyle=':', linewidth=1.5, label='Generative Performance')
        handles.append(generative_line)
        labels.append('Generative Performance')
        
        ax.legend(
            handles=handles,
            labels=labels,
            title="Model",
            loc="lower right",
            frameon=True,
            borderaxespad=0.5,
            ncol=n_cols,
            fontsize=11
        )
    else:
        ax.legend(
            title="Model",
            loc="lower right",
            frameon=True,
            borderaxespad=0.5,
            ncol=n_cols,
            fontsize=11
        )

    plt.tight_layout()
    return fig, used_color_map, used_generative_map, legend_labels


# ---------------------- Main ---------------------- #
def main(argv: List[str] | None = None) -> None:
    plt.rcParams.update({"font.size": 12})
    args = parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Optional colors and generative accuracy input
    input_colors = None
    generative_accs = None
    if args.colors_file:
        input_colors, generative_accs = read_colors_file(args.colors_file)

    # Decide output filenames and inputs
    if args.json_file:
        output_name = args.output_name or derive_output_name_from_file(args.json_file)
        json_files = [args.json_file]
    else:
        output_name = args.output_name or "layers_compare"
        json_files = find_jsons_in_dir(args.dir, args.glob)

    out_pdf = args.output_dir / f"{output_name}.pdf"
    out_png = args.output_dir / f"{output_name}.png"

    cutoff_spec = parse_layer_cutoff(args.layer_cutoff)

    print(f"Found {len(json_files)} file(s).")
    for f in json_files:
        print(f"  - {f.name}")

    fig, used_colors, used_generative, labels = plot_multiple(
        json_files, tuple(args.figsize), args.dpi, args.x_percent, cutoff_spec, 
        input_colors, generative_accs, args.show_generative
    )

    # Save figure(s)
    print(f"Saving to {out_pdf}")
    fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight")
    if not args.no_png:
        fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Save colors mapping if requested
    if args.save_colors:
        if isinstance(args.save_colors, str):
            colors_out = Path(args.save_colors)
        else:
            colors_out = args.output_dir / f"{output_name}_colors.txt"
        # Keep mapping order consistent with legend labels by rebuilding ordered dict
        ordered: Dict[str, str] = {}
        # labels are of the form "<base_label> (nL)" — map back to base_label
        for leg in labels:
            base = leg.split(" (", 1)[0]
            if base in used_colors and base not in ordered:
                ordered[base] = used_colors[base]
        # Include any remaining (unlikely) stray mappings
        for k, v in used_colors.items():
            if k not in ordered:
                ordered[k] = v
        save_colors_file(ordered, used_generative, colors_out)

    # Print a compact summary
    print("\nSummary:")
    print(f"  Mode: {'single file' if args.json_file else 'directory'}")
    print(f"  Output name: {output_name}")
    print(f"  X-axis: {'% of plotted depth (per model)' if args.x_percent else 'layer index'}")
    if cutoff_spec is None:
        print("  Cutoff: all layers")
    elif isinstance(cutoff_spec, int):
        print(f"  Cutoff: first {cutoff_spec} layers")
    else:
        print(f"  Cutoff: first {cutoff_spec:.0f}% of layers")
    if args.colors_file:
        print(f"  Colors input: {args.colors_file}")
    if args.show_generative:
        print("  Generative accuracy lines: enabled")
    if args.save_colors:
        print("  Colors saved.")


if __name__ == "__main__":
    main()