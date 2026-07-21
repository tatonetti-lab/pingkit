#!/usr/bin/env python
"""
layer_eval.py — Layer-by-layer CV sweep, evaluated per embedding part.

Sweeps every transformer layer for each specified part (e.g. attn, mlp, rs)
independently, trains a probe with k-fold cross-validation, and saves all
results to a single JSON keyed by part name.

Example:
    python layer_eval.py \
        --embedding_dir mmlu_answer \
        --labels_csv   mmlu_g.csv \
        --label_col    answer \
        --parts        rs,attn,mlp \
        --metric       accuracy \
        --model_type   mlp \
        --k_folds      5 \
        --restack      true \
        --output       my_results.json
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

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from pingkit.extraction import extract_token_vectors
from pingkit.model import fit, load_npz_features


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layer-by-layer CV sweep per part.")
    p.add_argument("--embedding_dir", required=True, type=Path,
                   help="Root directory containing rs/, attn/, mlp/ folders.")
    p.add_argument("--labels_csv", required=True, type=Path,
                   help="CSV with sample IDs and labels.")
    p.add_argument("--label_col", default="label",
                   help="Column name for ground-truth labels.")
    p.add_argument("--id_col", default="id",
                   help="Column for sample IDs (must match embedding filenames).")
    p.add_argument("--parts", default="rs",
                   help='Comma-separated parts to evaluate independently (e.g. "rs,attn,mlp").')
    p.add_argument("--metric", choices=["accuracy", "auc", "f1"], default="accuracy")
    p.add_argument("--model_type", choices=["mlp", "cnn"], default="mlp")
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--restack", type=str, default="true", choices=["true", "false"],
                   help="Whether to re-stack embeddings into NPZ for each layer.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON path. Auto-generated if not provided.")
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


_LAYER_RE = re.compile(r"_L(\d+)\.csv$")


def discover_layers(embedding_dir: Path, part: str) -> List[int]:
    """Return sorted layer indices found in embedding_dir/<part>/."""
    part_dir = embedding_dir / part
    if not part_dir.exists():
        raise FileNotFoundError(f"{part_dir} does not exist")
    layers = set()
    for f in part_dir.iterdir():
        m = _LAYER_RE.search(f.name)
        if m:
            layers.add(int(m.group(1)))
    if not layers:
        raise RuntimeError(f"No *_L<layer>.csv files found in {part_dir}")
    return sorted(layers)


def metric_fn_factory(name: str, num_classes: int) -> Callable:
    """Return scorer: (y_true, y_pred_proba) -> float."""
    if name == "accuracy":
        return lambda yt, yp: accuracy_score(yt, yp.argmax(1))
    elif name == "f1":
        return lambda yt, yp: f1_score(yt, yp.argmax(1), average="macro")
    elif name == "auc":
        def _auc(yt, yp):
            if num_classes == 2:
                return roc_auc_score(yt, yp[:, 1])
            return roc_auc_score(yt, yp, multi_class="ovr", average="macro")
        return _auc
    raise ValueError(f"Unknown metric {name}")


def stack_features(embedding_dir: Path, layer: int, part: str, n_jobs: int = 8) -> Path:
    """Stack CSVs for one part+layer into a single NPZ, overwriting any existing NPZ."""
    out = embedding_dir / "condensed" / f"features_{part}_L{layer}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    extract_token_vectors(
        embedding_dir=str(embedding_dir),
        parts=[part],
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
    """Run k-fold CV, return (mean, min, max) of metric."""
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=101)
    num_classes = len(np.unique(y))
    scorer = metric_fn_factory(metric_name, num_classes)
    vals: List[float] = []

    for train_idx, test_idx in splitter.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx].values
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx].values

        model, _ = fit(
            X_train, y_train,
            model_type=model_type,
            meta=meta if model_type == "cnn" else None,
            num_classes=num_classes,
            metric="loss",
            device=device,
            **fit_kwargs,
        )

        model.eval()
        with torch.no_grad():
            feats = torch.tensor(X_test.values.astype(np.float32), device=device)
            logits = model(feats) if model_type == "mlp" else model(feats)[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        vals.append(scorer(y_test, probs))

    arr = np.array(vals)
    return float(arr.mean()), float(arr.min()), float(arr.max())


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]

    # Load labels
    df_labels = pd.read_csv(args.labels_csv)
    if args.id_col not in df_labels.columns or args.label_col not in df_labels.columns:
        sys.exit(f"{args.labels_csv} must contain columns [{args.id_col}, {args.label_col}]")

    y_series = df_labels.set_index(args.id_col)[args.label_col]
    label_encoder = LabelEncoder()
    if not pd.api.types.is_numeric_dtype(y_series):
        print("Detected categorical labels. Encoding to numeric...")
        y_series = pd.Series(
            label_encoder.fit_transform(y_series),
            index=y_series.index, name=y_series.name,
        )
        print(f"Label mapping: {dict(enumerate(label_encoder.classes_))}")

    fit_kwargs = dict(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=0.15,
        early_stopping=True,
        patience=10,
        random_state=101,
    )

    # Results keyed by part, then layer
    all_results: dict = {}

    for part in parts:
        print(f"\n{'='*40}")
        print(f"Part: {part}")
        print(f"{'='*40}")

        layers = discover_layers(args.embedding_dir, part)
        print(f"Found layers: {layers}")
        part_results: dict = {}

        for layer in layers:
            print(f"\n  --- Layer {layer} ---")

            if args.restack.lower() == "true":
                npz_path = stack_features(args.embedding_dir, layer, part)
            else:
                npz_path = args.embedding_dir / "condensed" / f"features_{part}_L{layer}.npz"
                if not npz_path.exists():
                    sys.exit(f"{npz_path} missing; rerun with --restack true.")

            X_df, meta = load_npz_features(str(npz_path))

            # Align with labels
            common = X_df.index.intersection(y_series.index)
            if len(common) < len(y_series):
                print(f"  Warning: {len(y_series) - len(common)} samples missing embeddings")
            X_df = X_df.loc[common]
            y = y_series.loc[common]

            mean_m, min_m, max_m = kfold_cv(
                X_df, y,
                k=args.k_folds,
                model_type=args.model_type,
                meta=meta,
                metric_name=args.metric,
                fit_kwargs=fit_kwargs,
                device=args.device,
            )

            part_results[layer] = {"mean": mean_m, "min": min_m, "max": max_m}
            print(f"  {args.metric}: {mean_m:.4f}  (min={min_m:.4f}, max={max_m:.4f})")

        all_results[part] = part_results

    # Save
    if args.output is not None:
        json_path = args.output
        json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = args.labels_csv.stem
        json_path = args.embedding_dir / f"Layer_eval_{stem}_{timestamp}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)

    args_serialisable = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    with open(json_path, "w") as f:
        json.dump({"args": args_serialisable, "results": all_results}, f, indent=2)

    print(f"\nSaved metrics -> {json_path}")


if __name__ == "__main__":
    main()