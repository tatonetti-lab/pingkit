from __future__ import annotations
import os
import json
import logging
import time
from typing import Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)

LOGGER = logging.getLogger(__name__)
__all__ = [
    "PlugClassifier",
    "PlugContrastiveCNN",
    # training helpers
    "cross_validate", "fit",
    # I/O helpers
    "save_artifacts", "load_artifacts", "predict",
    "load_npz_features",
]


class PlugClassifier(nn.Module):
    # Three‑layer residual MLP whose width is chosen so that the total
    # number of trainable parameters is **approximately
    # target_ratio × n_examples.**
    #
    # For tabular data a target_ratio of 3‑10 is usually plenty; the
    # default 5 keeps capacity modest even for small data sets.
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        *,
        n_examples: int | None = None,
        target_ratio: float = 5.0,      # max desired (params / samples)
        p_drop: float = 0.3,
        out_floor: int = 16,
        width_cap: int = 128,           # hard upper bound for fc1
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Determine fc1 width ------------------------------------------------
        # ------------------------------------------------------------------
        if n_examples is None:
            # Inference‑time path: fall back to a tiny network
            fc1_w = 32
        else:
            # Solve   params ≈ input_dim * fc1   ⇒  fc1 ≈ target·N / d
            est_fc1 = int(target_ratio * n_examples / input_dim)
            fc1_w = int(np.clip(est_fc1, 16, width_cap))

        fc2_w = max(16, fc1_w // 2)
        out_w = max(out_floor, fc2_w // 4)

        # Helper for a dense block
        def block(i, o):
            return nn.Sequential(
                nn.Linear(i, o),
                nn.BatchNorm1d(o),
                nn.ReLU(),
                nn.Dropout(p_drop),
            )

        # Layers
        self.fc1 = block(input_dim, fc1_w)
        self.fc2 = block(fc1_w, fc2_w)
        self.res = nn.Sequential(block(fc2_w, fc2_w),
                                 nn.Linear(fc2_w, fc2_w))   # residual
        self.out = nn.Sequential(block(fc2_w, out_w),
                                 nn.Linear(out_w, num_classes))

        # Log capacity
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        LOGGER.info(
            "PlugClassifier • d=%d N=%s target=%.1f ⇒ fc1=%d params=%d (≈%.1f×N)",
            input_dim, n_examples, target_ratio, fc1_w,
            n_params, n_params / (n_examples or 1)
        )

    def forward(self, x):               # x: (B, d)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x + self.res(x)
        return self.out(x)              # logits (B, C)



class PlugCNNEncoder(nn.Module):
    # Capacity‑aware CNN encoder.  Stores self.proj_dim so the caller can
    # build a classifier without poking into the backbone.
    def __init__(
        self,
        n_parts: int,
        n_layers: int,
        hidden: int,
        *,
        n_examples: int | None = None,
        target_ratio: float = 5.0,
        width_cap: int = 64,
        proj_mult: int = 2,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.use_1d = (n_parts == 1)
        in_ch = n_layers

        # ---- choose base width -------------------------------------------------
        if n_examples is None:
            base = 16
        else:
            base = int(np.clip(np.sqrt(target_ratio * n_examples / 10), 8, width_cap))

        c1 = base
        c2 = base * 2
        self.proj_dim = max(32, base * proj_mult)   # <‑‑ store here

        self.layer_norm = nn.LayerNorm(hidden)

        # ---- backbone ----------------------------------------------------------
        if self.use_1d:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_ch, c1, 5, padding=2), nn.BatchNorm1d(c1), nn.ReLU(),
                nn.Conv1d(c1, c2, 3, padding=1),    nn.BatchNorm1d(c2), nn.ReLU(),
                nn.Conv1d(c2, self.proj_dim, 1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(p_drop),
            )
        else:
            k_h = 3 if n_parts > 2 else n_parts
            self.backbone = nn.Sequential(
                nn.Conv2d(in_ch, c1, (k_h, 5), padding=(k_h // 2, 2)),
                nn.BatchNorm2d(c1), nn.ReLU(),
                nn.Conv2d(c1, c2, 1), nn.BatchNorm2d(c2), nn.ReLU(),
                nn.Conv2d(c2, self.proj_dim, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p_drop),
            )

        LOGGER.info("CNN encoder  base=%d  proj_dim=%d", base, self.proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        return self.backbone(x)           # (B, proj_dim)




class PlugContrastiveCNN(nn.Module):
    # CNN encoder + small MLP head.  Width scales via target_ratio.
    def __init__(
        self,
        n_parts: int,
        n_layers: int,
        hidden: int,
        *,
        num_classes: int = 2,
        n_examples: int | None = None,
        target_ratio: float = 5.0,
        p_drop: float = 0.1,
        width_cap: int = 64,
        proj_mult: int = 2,
    ):
        super().__init__()
        self.n_parts = n_parts
        self.n_layers = n_layers
        self.hidden = hidden

        self.encoder = PlugCNNEncoder(
            n_parts,
            n_layers,
            hidden,
            n_examples=n_examples,
            target_ratio=target_ratio,
            width_cap=width_cap,
            proj_mult=proj_mult,
            p_drop=p_drop,
        )
        proj_dim = self.encoder.proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(proj_dim, max(32, proj_dim // 4)), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(max(32, proj_dim // 4), num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        LOGGER.info(
            "PlugContrastiveCNN • classes=%d  params=%d (≈%.1f×N)",
            num_classes, n_params, n_params / (n_examples or 1)
        )

    # flatten grid -> encode -> logits (+ embedding)
    def forward(self, flat_x):
        b = flat_x.size(0)
        if self.n_parts == 1:
            x = flat_x.view(b, self.n_layers, self.hidden)
        else:
            x = flat_x.view(b, self.n_layers, self.n_parts, self.hidden)
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z



# SUPERVISED CONTRASTIVE LOSS

class SupConLoss(nn.Module):
    # Supervised Contrastive Loss from Khosla et al. (2020)
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature
        self.eps = 1e-8

    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        f = torch.nn.functional.normalize(feats, p=2, dim=1)
        sim = torch.matmul(f, f.T) / self.tau
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
        sim = sim * logits_mask
        exp = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp.sum(1, keepdim=True) + self.eps)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)
        return -mean_log_prob_pos.mean()


# I/O UTILITIES

def load_npz_features(npz_path: str) -> Tuple[pd.DataFrame, dict]:
    # Load features saved as .npz and return DataFrame + meta.
    with np.load(npz_path, allow_pickle=True) as npz:
        data   = npz["data"]          # (feat, samples)
        ids    = npz["columns"]
        parts  = list(npz["parts"])
        layers = list(npz["layers"])
        hidden = int(npz["hidden_size"][0])
    df   = pd.DataFrame(data.T, index=ids)
    meta = {"parts": parts, "layers": layers, "hidden_size": hidden}
    return df, meta


def _to_numpy(X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    # Convert DataFrame/array to (float32 np.ndarray, id_array)
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float32), X.index.to_numpy()
    return X.astype(np.float32), np.arange(X.shape[0])


def _make_model(
    model_type: str,
    *,
    input_dim: int,
    meta: dict | None,
    num_classes: int = 2,
    n_examples: int | None = None,
    target_ratio: float = 5.0,
    p_drop: float = 0.3,
):
    if model_type == "mlp":
        return PlugClassifier(
            input_dim,
            num_classes,
            n_examples=n_examples,
            target_ratio=target_ratio,
            p_drop=p_drop,
        )
    if model_type == "cnn":
        if meta is None:
            raise ValueError("meta required for CNN.")
        return PlugContrastiveCNN(
            len(meta["parts"]),
            len(meta["layers"]),
            meta["hidden_size"],
            num_classes=num_classes,
            n_examples=n_examples,          # <‑‑ now forwarded
            target_ratio=target_ratio,      # <‑‑ now forwarded
            p_drop=p_drop,
        )
    raise ValueError(f"Unknown model_type {model_type!r}")





def _forward(
    model: nn.Module,
    xb: torch.Tensor,
    model_type: str
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    # Forward helper that always returns (probabilities, embedding_or_None)
    if model_type == "cnn":
        logits, z = model(xb)
    else:
        logits = model(xb)
        z = None
    probs = torch.softmax(logits, dim=1)
    return probs, z


def _path_prefix(path: str) -> str:
    # Strip weight‑file extension to obtain the base prefix.
    root, ext = os.path.splitext(path)
    return root if ext.lower() in {".pt", ".pth", ".bin"} else path


def _json_default(obj):
    # Make numpy scalars / arrays JSON‑serialisable.
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj)} cannot be JSON‑encoded")


def save_artifacts(
    model: torch.nn.Module,
    *,
    path: str = "artifacts/plug",
    meta: dict | None = None,
) -> Tuple[str, str]:
    # Save model weights + meta and return absolute file paths.
    prefix = _path_prefix(path)
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    # Detect model type and number of classes
    if isinstance(model, PlugClassifier):
        mtype = "mlp"
        num_classes = model.out[-1].out_features
    elif isinstance(model, PlugContrastiveCNN):
        mtype = "cnn"
        num_classes = model.classifier[-1].out_features
    else:
        raise TypeError(f"Unsupported model class {type(model)}")

    meta_out: dict = {"model_type": mtype, "num_classes": num_classes}
    if meta:
        meta_out.update(meta)

    if mtype == "mlp":
        meta_out.setdefault("input_dim", int(model.fc1[0].in_features))
    else:
        meta_out.setdefault("parts",       model.n_parts)
        meta_out.setdefault("layers",      model.n_layers)
        meta_out.setdefault("hidden_size", model.hidden)

    weights_path = prefix + ".pt"
    torch.save(model.state_dict(), weights_path)

    meta_path = prefix + ".json"
    with open(meta_path, "w") as fp:
        json.dump(meta_out, fp, indent=2, default=_json_default)

    LOGGER.info("Artifacts saved → %s  (+ meta %s)", weights_path, meta_path)
    return os.path.abspath(weights_path), os.path.abspath(meta_path)


def _build_from_meta(meta: dict, device: Union[str, torch.device] = "cpu") -> torch.nn.Module:
    # Recreate uninitialised model from stored metadata.
    num_classes = int(meta.get("num_classes", 2))
    mtype = meta.get("model_type", "mlp")
    if mtype == "mlp":
        return PlugClassifier(int(meta["input_dim"]), num_classes).to(device)
    if mtype == "cnn":
        return _make_model(
            "cnn", input_dim=-1, meta=meta,
            num_classes=num_classes
        ).to(device)
    raise ValueError(f"Unknown model_type {mtype!r}")


def load_artifacts(
    path: str,
    *,
    device: str | torch.device = "cpu",
) -> Tuple[torch.nn.Module, dict]:
    # Load model weights + meta (generated by save_artifacts).
    prefix = _path_prefix(path)
    weights_path = None
    for ext in (".pt", ".pth"):
        cand = prefix + ext
        if os.path.isfile(cand):
            weights_path = cand
            break
    if weights_path is None:
        raise FileNotFoundError(f"No weight file (.pt or .pth) for prefix '{prefix}'.")

    meta_path = prefix + ".json"
    if os.path.isfile(meta_path):
        with open(meta_path) as fp:
            meta = json.load(fp)
    else:
        raise FileNotFoundError("meta.json missing; cannot rebuild model.")

    device = torch.device(device)
    model = _build_from_meta(meta, device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    LOGGER.info("Loaded model ← %s  (device=%s)", weights_path, device)
    return model, meta


def _read_features(src: Union[str, pd.DataFrame, np.ndarray]) -> Tuple[pd.DataFrame, dict | None]:
    # Convenience loader that accepts DataFrame, ndarray, .csv, or .npz
    if isinstance(src, pd.DataFrame):
        return src, None
    if isinstance(src, np.ndarray):
        return pd.DataFrame(src), None
    if isinstance(src, str):
        if src.endswith(".npz"):
            return load_npz_features(src)
        return pd.read_csv(src, index_col=0), None
    raise TypeError(type(src))


# CROSS‑VALIDATION LOOP

def cross_validate(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, pd.DataFrame, np.ndarray],
    *,
    model_type: str = "mlp",
    meta: dict | None = None,
    num_classes: int = 2,
    label_col: str | None = None,
    leave_out_col: str | None = None,
    groups: Union[pd.Series, np.ndarray, None] = None,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    n_splits: int = 5,
    batch_size: int = 128,
    device: str | torch.device = "cuda",
    eval_every: int = 1,
    contrastive_weight: float = 1.0,
    out_dir: str = "artifacts",
    run_name: str | None = None,
    early_stopping: bool = False,
    patience: int = 10,
) -> np.ndarray:
    # Cross‑validate multi‑class models.  Returns per‑sample probability matrix.
    from datetime import datetime
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_out_dir = os.path.join(out_dir, run_name)
    os.makedirs(full_out_dir, exist_ok=True)

    # Labels to ndarray (int class indices)
    if isinstance(y, pd.DataFrame):
        if label_col is None:
            raise ValueError("label_col required when y is a DataFrame.")
        y_vec = y[label_col].to_numpy(dtype=int)
    elif isinstance(y, pd.Series):
        y_vec = y.to_numpy(dtype=int)
    else:
        y_vec = np.asarray(y, dtype=int)

    # Group splitting if requested
    if groups is None and leave_out_col is not None:
        if isinstance(X, pd.DataFrame) and leave_out_col in X.columns:
            groups = X[leave_out_col]
        elif isinstance(y, pd.DataFrame) and leave_out_col in y.columns:
            groups = y[leave_out_col]
        else:
            raise KeyError(f"{leave_out_col!r} not found in X nor y.")
    if groups is not None:
        groups = np.asarray(groups)
        splitter = LeaveOneGroupOut()
        split_gen = splitter.split(X, y_vec, groups)
        n_splits = len(np.unique(groups))
        mode = f"LOGO ({n_splits} groups)"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=101)
        split_gen = splitter.split(X, y_vec)
        mode = f"{n_splits}-fold stratified"

    LOGGER.info(
        "CV • %s • model=%s • classes=%d • epochs=%d • run=%s",
        mode, model_type, num_classes, n_epochs, run_name
    )

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    X_np, _ = _to_numpy(X)

    ce_loss = nn.CrossEntropyLoss()
    supcon = SupConLoss().to(device)

    preds = np.zeros((len(y_vec), num_classes), np.float32)
    all_epochs: list[list[int]] = []
    all_metrics: list[list[float]] = []
    summary: list[dict] = []
    cv_start = time.time()

    for fold, (tr, va) in enumerate(split_gen, 1):
        grp_name = np.unique(groups[va])[0] if groups is not None else "N/A"
        LOGGER.info("Fold %d/%d (group=%s) – training …", fold, n_splits, grp_name)

        model = _make_model(
            model_type,
            input_dim=X_np.shape[1],
            meta=meta,
            num_classes=num_classes,
            n_examples=len(tr),               # <‑‑ add this
        ).to(device)

        opt = optim.Adam(model.parameters(), lr=learning_rate)
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X_np[tr]), torch.tensor(y_vec[tr], dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        epochs_checked: list[int] = []
        metric_hist: list[float] = []
        best_metric = -float("inf")
        epochs_no_imp = 0

        for ep in range(1, n_epochs + 1):
            model.train()
            running_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)

                # forward and loss
                if model_type == "cnn":
                    logits, z = model(xb)
                    cls_loss = ce_loss(logits, yb)
                    loss = cls_loss + contrastive_weight * supcon(z, yb)
                else:
                    logits = model(xb)
                    loss = ce_loss(logits, yb)

                loss.backward()
                opt.step()
                running_loss += loss.item() * xb.size(0)

            # periodic evaluation
            if ep % eval_every == 0 or ep == n_epochs:
                model.eval()
                with torch.no_grad():
                    p_val, _ = _forward(
                        model,
                        torch.tensor(X_np[va]).to(device),
                        model_type
                    )
                    p_tr, _ = _forward(
                        model,
                        torch.tensor(X_np[tr]).to(device),
                        model_type
                    )

                v = p_val.cpu().numpy()
                t = p_tr.cpu().numpy()

                # Macro ROC‑AUC over classes
                if num_classes == 2:
                    # binary → use the positive‑class probability
                    val_auc = roc_auc_score(y_vec[va], v[:, 1])
                    tr_auc  = roc_auc_score(y_vec[tr], t[:, 1])
                else:
                    # multi‑class → macro One‑Vs‑Rest
                    val_auc = roc_auc_score(y_vec[va], v, multi_class="ovr", average="macro")
                    tr_auc  = roc_auc_score(y_vec[tr], t, multi_class="ovr", average="macro")
                LOGGER.info("Fold %d ep %3d | train_auc %.4f val_auc %.4f",
                            fold, ep, tr_auc, val_auc)

                epochs_checked.append(ep)
                metric_hist.append(val_auc)
                p_val_final = v

                # early stopping logic
                if early_stopping:
                    if val_auc > best_metric:
                        best_metric = val_auc
                        epochs_no_imp = 0
                    else:
                        epochs_no_imp += 1
                        if epochs_no_imp >= patience:
                            LOGGER.info("Early stopping after %d epochs", patience)
                            break

        preds[va] = p_val_final
        all_epochs.append(epochs_checked)
        all_metrics.append(metric_hist)
        summary.append({
            "fold": fold,
            "group": str(grp_name),
            "final_auc": metric_hist[-1],
            "epoch_final": epochs_checked[-1]
        })
        LOGGER.info("Fold %d done – final_auc %.4f @ epoch %d",
                    fold, metric_hist[-1], epochs_checked[-1])

    # Plot ROC‑AUC curves
    plt.figure()
    min_v = min(min(m) for m in all_metrics)
    for f, (eps, mets) in enumerate(zip(all_epochs, all_metrics), 1):
        plt.scatter(eps, mets, s=15, alpha=0.3)
        plt.plot(eps, pd.Series(mets).rolling(3, min_periods=1).mean(),
                 label=f"fold {f}")
    plt.ylabel("Macro ROC‑AUC")
    plt.ylim(min_v - 0.05, 1.0)
    plt.xlabel("epoch")
    plt.title("Cross‑validation ROC‑AUC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(full_out_dir, "cv_curves.png")
    plt.savefig(png_path, dpi=120)
    plt.close()

    # Summary JSON
    elapsed = time.time() - cv_start
    if num_classes == 2:
        overall_auc = roc_auc_score(y_vec, preds[:, 1])
    else:
        overall_auc = roc_auc_score(
            y_vec, preds, multi_class="ovr", average="macro"
        )
    LOGGER.info("CV done – overall macro AUC %.4f | %.1fs", overall_auc, elapsed)
    summary_dict = {
        "overall_auc": overall_auc,
        "sec_total": elapsed,
        "folds": summary,
        "curves_png": os.path.basename(png_path)
    }
    with open(os.path.join(full_out_dir, "cv_summary.json"), "w") as fp:
        json.dump(summary_dict, fp, indent=2)

    return preds


# FULL‑DATA TRAINING


def fit(
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    *,
    model_type: str = "mlp",
    meta: dict | None = None,
    num_classes: int = 2,
    n_epochs: int = 300,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda",
    contrastive_weight: float = 1.0,
    early_stopping: bool = False,
    patience: int = 10
):
    # Train on full dataset and return the fitted model.
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    X_np, _ = _to_numpy(X)
    y_np = np.asarray(y, dtype=int)

    model = _make_model(
        model_type,
        input_dim=X_np.shape[1],
        meta=meta,
        num_classes=num_classes,
        n_examples=X_np.shape[0],         # <‑‑ add this
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=learning_rate)
    supcon = SupConLoss().to(device)
    ce_loss = nn.CrossEntropyLoss()

    ds = torch.utils.data.TensorDataset(
        torch.tensor(X_np), torch.tensor(y_np, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    no_imp = 0

    for ep in tqdm(range(n_epochs), desc="Full‑fit", ncols=80):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)

            if model_type == "cnn":
                logits, z = model(xb)
                loss = ce_loss(logits, yb) + contrastive_weight * supcon(z, yb)
            else:
                logits = model(xb)
                loss = ce_loss(logits, yb)

            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(ds)
        if early_stopping:
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    LOGGER.info("Early stopping after %d epochs", patience)
                    break

    model.eval()
    return model


# PREDICTION + OPTIONAL EVALUATION

def predict(
    features: Union[str, pd.DataFrame, np.ndarray],
    *,
    model_path: str,
    output_csv: str = "predictions.csv",
    response_csv: str | None = None,
    response_col: str = "answer",
    device: str = "cuda",
) -> pd.DataFrame:
    # 1.  Load feature matrix (csv / npz / ndarray / DataFrame)
    X_df, _ = _read_features(features)
    ids = X_df.index.to_numpy()
    X_np = X_df.values.astype(np.float32)

    # 2.  Re‑instantiate model from saved artefacts
    model, meta = load_artifacts(model_path, device=device)
    model_type  = meta.get("model_type", "mlp")
    num_classes = int(meta.get("num_classes", 2))

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        probs_tensor, _ = _forward(
            model,
            torch.tensor(X_np, device=device),
            model_type,
        )
    probs = probs_tensor.cpu().numpy()        # (N, C) soft‑max probs

    # 3.  Build prediction DataFrame
    if num_classes == 2:
        # Binary → store only the positive‑class probability for convenience
        pred_df = pd.DataFrame(
            {"id": ids, "prob_positive": probs[:, 1]},
            columns=["id", "prob_positive"],
        )
    else:
        # Multi‑class → one probability column per class
        pred_cols = {f"prob_class_{c}": probs[:, c] for c in range(num_classes)}
        pred_df = pd.DataFrame({"id": ids, **pred_cols})

    pred_df.to_csv(output_csv, index=False)
    LOGGER.info("Predictions → %s", output_csv)

    # 4.  Optional evaluation against ground‑truth CSV
    if response_csv is not None:
        gt_df  = pd.read_csv(response_csv, index_col="id")
        merged = gt_df[[response_col]].join(pred_df.set_index("id"), how="inner")
        if merged.empty:
            LOGGER.warning("No overlapping IDs with ground‑truth; metrics skipped.")
        else:
            y_true = merged[response_col].values.astype(int)

            if num_classes == 2:
                # Binary metrics
                y_pred_prob = merged["prob_positive"].values
                y_pred_cls  = (y_pred_prob >= 0.5).astype(int)
                auc = roc_auc_score(y_true, y_pred_prob)
            else:
                # Multi‑class metrics
                prob_cols   = [f"prob_class_{c}" for c in range(num_classes)]
                y_pred_prob = merged[prob_cols].values
                y_pred_cls  = y_pred_prob.argmax(1)
                auc = roc_auc_score(
                    y_true, y_pred_prob, multi_class="ovr", average="macro"
                )

            metrics = {
                "accuracy":  accuracy_score(y_true, y_pred_cls),
                "macro_f1":  f1_score(y_true, y_pred_cls, average="macro"),
                "auc":       auc,
                "precision_macro": precision_score(y_true, y_pred_cls, average="macro"),
                "recall_macro":    recall_score(y_true, y_pred_cls, average="macro"),
            }

            mpath = Path(output_csv).with_suffix(".metrics.json")
            with open(mpath, "w") as fp:
                json.dump(metrics, fp, indent=2)
            LOGGER.info("Metrics → %s", mpath)

    return pred_df