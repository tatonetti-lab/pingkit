#!/usr/bin/env python
"""
layer_eval.py
=============

Line‑sweeps every transformer *layer* in a PingKit embedding directory,
trains a probe with **k‑fold cross‑validation**, and reports the metric
(mean ± min/max across folds) so we can spot the "sweet‑spot" layer.

-----------------------------------------------------------------------
Example
-------
$ python layer_eval.py \
      --embedding_dir mmlu_answer \
      --labels_csv   mmlu_g.csv \
      --label_col    answer \
      --parts        rs \
      --metric       accuracy \
      --model_type   mlp \
      --k_folds      5 \
      --restack      true \
      --output       my_results.json
-----------------------------------------------------------------------

"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
# ----------------------------- pingkit -------------------------------- #
from pingkit.extraction import extract_token_vectors
from pingkit.model import fit, load_npz_features

# --------------------------- cli helpers ------------------------------ #


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(description="Layer‑by‑layer CV sweep.")

    p.add_argument(
        "--embedding_dir",
        required=True,
        type=Path,
        help="Root directory containing rs/, attn/, mlp/ folders of per‑token CSVs.",
    )
    p.add_argument(
        "--labels_csv",
        required=True,
        type=Path,
        help="CSV with sample IDs and labels.",
    )
    p.add_argument(
        "--label_col",
        default="label",
        help="Column name in labels_csv containing ground‑truth labels.",
    )
    p.add_argument(
        "--id_col",
        default="id",
        help="Column in labels_csv that holds the sample IDs (must match embedding filenames).",
    )
    p.add_argument(
        "--parts",
        default="rs",
        help='Comma‑separated list of embedding parts to use (e.g. "rs", "rs,attn").',
    )
    p.add_argument(
        "--metric",
        choices=["accuracy", "auc", "f1"],
        default="accuracy",
        help="Performance metric to maximise.",
    )
    p.add_argument(
        "--model_type",
        choices=["mlp", "cnn"],
        default="mlp",
        help="Probe architecture to train.",
    )
    p.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of CV folds.",
    )
    p.add_argument(
        "--restack",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to *re‑stack* (concatenate) embeddings into an NPZ for each layer.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path. If not provided, auto-generates filename based on timestamp and input CSV.",
    )
    p.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Maximum epochs for each fold.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Mini‑batch size for training.",
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Torch device string ("cuda" or "cpu").',
    )

    return p.parse_args(argv)


# -------------------------- utility funcs ----------------------------- #


_LAYER_RE = re.compile(r"_L(\d+)\.csv$")


def discover_layers(embedding_dir: Path) -> List[int]:
    """Return all layer indices present in embedding_dir/rs/."""
    rs_dir = embedding_dir / "rs"
    if not rs_dir.exists():
        raise FileNotFoundError(f"{rs_dir} does not exist")
    layers = set()
    for file in rs_dir.iterdir():
        m = _LAYER_RE.search(file.name)
        if m:
            layers.add(int(m.group(1)))
    if not layers:
        raise RuntimeError("No *_L<layer>.csv files found in rs/")
    return sorted(layers)


def metric_fn_factory(name: str, num_classes: int) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return a scorer taking (y_true, y_pred_proba) -> float."""
    if name == "accuracy":

        def _fn(y_true, y_prob):
            return accuracy_score(y_true, y_prob.argmax(1))

    elif name == "f1":

        def _fn(y_true, y_prob):
            return f1_score(y_true, y_prob.argmax(1), average="macro")

    elif name == "auc":

        def _fn(y_true, y_prob):
            if num_classes == 2:
                return roc_auc_score(y_true, y_prob[:, 1])
            return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

    else:
        raise ValueError(f"Unknown metric {name}")

    return _fn


def stack_features(
    embedding_dir: Path,
    layer: int,
    parts: List[str],
    n_jobs: int = 8,
) -> Path:
    """Stack csvs for `layer` into a single NPZ and return its path."""
    out = (
        embedding_dir
        / "condensed"
        / f"features_{'-'.join(parts)}_L{layer}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    # Only regenerate if missing
    if not out.exists():
        extract_token_vectors(
            embedding_dir=str(embedding_dir),
            parts=parts,
            layers=layer,
            output_file=str(out.with_suffix("")),
            save_csv=False,
            n_jobs=n_jobs,
        )
    return out


def kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    model_type: str,
    meta: dict,
    metric_name: str,
    fit_kwargs: dict,
    device: str,
) -> Tuple[float, float, float]:
    """Run k‑fold CV and return (mean, min, max) of metric across folds."""
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    metric_vals: List[float] = []
    scorer = metric_fn_factory(metric_name, num_classes=len(np.unique(y)))

    for train_idx, test_idx in splitter.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx].values
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx].values

        model, _ = fit(
            X_train,
            y_train,
            model_type=model_type,
            meta=meta if model_type == "cnn" else None,
            num_classes=len(np.unique(y)),
            metric="loss",  # we'll compute our own metric later
            device=device,
            **fit_kwargs,
        )

        # inference
        model.eval()
        with torch.no_grad():
            feats = torch.tensor(
                X_test.values.astype(np.float32), device=device
            )
            logits = model(feats) if model_type == "mlp" else model(feats)[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        metric_vals.append(scorer(y_test, probs))

    metric_vals = np.array(metric_vals)
    return float(metric_vals.mean()), float(metric_vals.min()), float(metric_vals.max())


def compute_auc(y_true, probs):
    """
    ROC AUC:
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
        # e.g., degenerate bootstrap where only one class is present (shouldn't
        # happen with stratified resampling, but safe-guard anyway)
        return float("nan")
        
# ------------------------------- main --------------------------------- #


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    layers = discover_layers(args.embedding_dir)
    print(f"Found layers: {layers}")

    # Load label file
    df_labels = pd.read_csv(args.labels_csv)
    if args.id_col not in df_labels.columns or args.label_col not in df_labels.columns:
        sys.exit(f"{args.labels_csv} must contain columns [{args.id_col}, {args.label_col}]")
    y_series = df_labels.set_index(args.id_col)[args.label_col]

    label_encoder = LabelEncoder()
    if y_series.dtype == 'object' or not np.issubdtype(y_series.dtype, np.number):
        print(f"Detected categorical labels. Encoding to numeric...")
        y_series = pd.Series(
            label_encoder.fit_transform(y_series),
            index=y_series.index,
            name=y_series.name
        )
        print(f"Label mapping: {dict(enumerate(label_encoder.classes_))}")

    results = defaultdict(dict)

    fit_kwargs = dict(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=0.15,
        early_stopping=True,
        patience=25,
        random_state=42,
    )

    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        if args.restack.lower() == "true":
            print("  stacking features...")
            npz_path = stack_features(
                embedding_dir=args.embedding_dir,
                layer=layer,
                parts=parts,
            )
        else:
            npz_path = (
                args.embedding_dir
                / "condensed"
                / f"features_{'-'.join(parts)}_L{layer}.npz"
            )
            if not npz_path.exists():
                sys.exit(f"{npz_path} missing; rerun with --restack true first.")

        X_df, meta = load_npz_features(str(npz_path))
        # Align with labels
        common = X_df.index.intersection(y_series.index)
        if len(common) < len(y_series):
            print(
                f"  Warning: {len(y_series) - len(common)} samples missing embeddings, "
                "dropping from evaluation."
            )
        X_df = X_df.loc[common]
        y = y_series.loc[common]

        mean_m, min_m, max_m = kfold_cv(
            X_df,
            y,
            k=args.k_folds,
            model_type=args.model_type,
            meta=meta,
            metric_name=args.metric,
            fit_kwargs=fit_kwargs,
            device=args.device,
        )

        results[layer] = {
            "mean": mean_m,
            "min": min_m,
            "max": max_m,
        }
        print(f"  {args.metric}: {mean_m:.4f}  (min={min_m:.4f}, max={max_m:.4f})")

    # ---------------- save artefacts ---------------- #
    
    # Determine output path
    if args.output is not None:
        # Use user-specified output path
        json_path = args.output
        # Ensure the output directory exists
        json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Use auto-generated filename (original behavior)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = args.labels_csv.stem
        out_dir = args.embedding_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"Layer_eval_{stem}_{timestamp}.json"

    args_serialisable = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }

    with open(json_path, "w") as f:
        json.dump(
            {
                "args": args_serialisable,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nSaved metrics → {json_path}")


if __name__ == "__main__":
    main()