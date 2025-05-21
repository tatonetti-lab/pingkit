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
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score
)

LOGGER = logging.getLogger(__name__)
__all__ = [
    # models
    "PlugClassifier", "PlugCNNClassifier", "PlugContrastiveCNN",
    # training helpers
    "cross_validate", "fit",
    # I/O helpers
    "save_artifacts", "load_artifacts", "predict",
    "load_npz_features"
]

# MODEL DEFINITIONS

class PlugClassifier(nn.Module):
    """
    Residual 3‑layer MLP whose width is a simple, interpretable
    function of *input_dim*.  Outputs a raw scalar; we apply the
    sigmoid externally for binary classification.
    """
    def __init__(self, input_dim: int, p_drop: float = 0.3):
        super().__init__()
        fc1_w = int(np.clip(4 * input_dim, 128, 1024))
        fc2_w = fc1_w // 2
        out_w = max(64, fc2_w // 4)

        def block(i, o):
            return nn.Sequential(
                nn.Linear(i, o),
                nn.BatchNorm1d(o),
                nn.ReLU(),
                nn.Dropout(p_drop),
            )

        self.fc1 = block(input_dim, fc1_w)
        self.fc2 = block(fc1_w, fc2_w)
        self.res = nn.Sequential(block(fc2_w, fc2_w),
                                 nn.Linear(fc2_w, fc2_w))
        # no Sigmoid here:
        self.out = nn.Sequential(block(fc2_w, out_w),
                                 nn.Linear(out_w, 1))

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        layer_shapes = (
            f"fc1: {input_dim}×{fc1_w}, "
            f"fc2: {fc1_w}×{fc2_w}, "
            f"out: {out_w}×1"
        )
        LOGGER.info(
            "PlugClassifier built • %s • total trainable params = %d",
            layer_shapes, n_params
        )

    def forward(self, x):         # x: (B, input_dim)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x + self.res(x)
        return self.out(x)        # (B,1)


class PlugCNNEncoder(nn.Module):
    """
    n_parts == 1 → Conv1d on shape (B, n_layers, hidden)
    else          → Conv2d on shape (B, n_layers, n_parts, hidden)

    - Per‑layer LayerNorm over hidden dimension
    - Two‑stage convolution (kernel 5→3 for 1‑D, or height k_h and width 5 for 2‑D)
    - Projection into *proj_dim* (default 128)
    - Global adaptive pooling to a single vector
    """
    def __init__(
        self,
        n_parts: int,
        n_layers: int,
        hidden: int,
        p_drop: float = 0.1,
        proj_dim: int = 128,
        w_red: int | None = None,
    ):
        super().__init__()
        self.use_1d = (n_parts == 1)
        in_ch = n_layers

        c1 = max(128, 4 * in_ch)
        c2 = max(256, 2 * c1)

        self.layer_norm = nn.LayerNorm(hidden)

        if self.use_1d:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_ch, c1, kernel_size=5, padding=2),
                nn.BatchNorm1d(c1),
                nn.ReLU(),
                nn.Conv1d(c1, c2, kernel_size=3, padding=1),
                nn.BatchNorm1d(c2),
                nn.ReLU(),
                nn.Conv1d(c2, proj_dim, kernel_size=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(p_drop),
            )
            LOGGER.info(
                "1‑D encoder %d→%d→%d | hidden %d→1",
                in_ch, c1, c2, hidden
            )
        else:
            k_h = 3 if n_parts > 2 else n_parts
            self.backbone = nn.Sequential(
                nn.Conv2d(in_ch, c1, kernel_size=(k_h, 5), padding=(k_h // 2, 2)),
                nn.BatchNorm2d(c1),
                nn.ReLU(),
                nn.Conv2d(c1, c2, kernel_size=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(),
                nn.Conv2d(c2, proj_dim, kernel_size=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p_drop),
            )
            LOGGER.info(
                "2‑D encoder %d→%d→%d | grid %d×%d→1",
                in_ch, c1, c2, n_parts, hidden
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        return self.backbone(x)


class PlugContrastiveCNN(nn.Module):
    """
    CNN encoder + small MLP classifier head.
    Final layer emits a raw logit (no sigmoid).
    """
    def __init__(
        self,
        n_parts: int,
        n_layers: int,
        hidden: int,
        p_drop: float = 0.1,
        proj_dim: int = 128
    ):
        super().__init__()
        self.n_parts = n_parts
        self.n_layers = n_layers
        self.hidden = hidden

        self.encoder = PlugCNNEncoder(
            n_parts, n_layers, hidden,
            p_drop=p_drop, proj_dim=proj_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(proj_dim, 64),       nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, 1)
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        LOGGER.info("PlugContrastiveCNN built • params=%d", n_params)

    def forward(self, flat_x):
        b = flat_x.size(0)
        if self.n_parts == 1:
            x = flat_x.view(b, self.n_layers, self.hidden)
        else:
            x = flat_x.view(b, self.n_layers, self.n_parts, self.hidden)
        z = self.encoder(x)
        logits = self.classifier(z).squeeze(1)
        return logits, z



# CONTRASTIVE LOSS

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
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


# HELPERS
def load_npz_features(npz_path: str) -> Tuple[pd.DataFrame, dict]:
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
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float32), X.index.to_numpy()
    return X.astype(np.float32), np.arange(X.shape[0])


def _make_model(
    model_type: str,
    *,
    input_dim: int,
    meta: dict | None,
    p_drop: float = 0.3
):
    if model_type == "mlp":
        return PlugClassifier(input_dim, p_drop)
    if model_type == "cnn":
        if meta is None:
            raise ValueError("meta required for CNN.")
        return PlugContrastiveCNN(
            len(meta["parts"]),
            len(meta["layers"]),
            meta["hidden_size"],
            p_drop=p_drop
        )
    raise ValueError(f"Unknown model_type {model_type!r}")


def _forward(
    model: nn.Module,
    xb: torch.Tensor,
    model_type: str
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    Return (probabilities, z) where probabilities are sigmoid(logits).
    """
    if model_type == "cnn":
        logits, z = model(xb)
    else:
        out = model(xb)              # (B,1)
        logits = out.squeeze(1)      # (B,)
        z = None

    probs = torch.sigmoid(logits)
    return probs, z


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) with equally‑spaced bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx  = np.digitize(probs, bins, right=True) - 1
    ece  = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        bin_acc  = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.mean()
    return float(ece)

def _path_prefix(path: str) -> str:
    """Return *path* without a weight‑file extension (.pt / .pth / .bin)."""
    root, ext = os.path.splitext(path)
    return root if ext.lower() in {".pt", ".pth", ".bin"} else path


def _json_default(obj):
    """Make numpy scalars / arrays JSON‑serialisable."""
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
    """Persist *weights* & *meta* side‑by‑side and return their absolute paths."""
    prefix = _path_prefix(path)
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    # Detect model type 
    if isinstance(model, PlugClassifier):
        mtype = "mlp"
    elif isinstance(model, PlugContrastiveCNN):
        mtype = "cnn"
    else:
        raise TypeError(f"Unsupported model class {type(model)}")

    # Assemble meta 
    meta_out: dict = {"model_type": mtype}
    if meta:
        meta_out.update(meta)

    if mtype == "mlp":
        meta_out.setdefault("input_dim", int(model.fc1[0].in_features))
    else:                                 # cnn
        meta_out.setdefault("parts",       model.n_parts)
        meta_out.setdefault("layers",      model.n_layers)
        meta_out.setdefault("hidden_size", model.hidden)

    # Write to disk 
    weights_path = prefix + ".pt"         # standardise on .pt when writing
    torch.save(model.state_dict(), weights_path)

    meta_path = prefix + ".json"
    with open(meta_path, "w") as fp:
        json.dump(meta_out, fp, indent=2, default=_json_default)

    LOGGER.info("Artifacts saved → %s  (+ meta %s)", weights_path, meta_path)
    return os.path.abspath(weights_path), os.path.abspath(meta_path)


def _build_from_meta(meta: dict, device: Union[str, torch.device] = "cpu") -> torch.nn.Module:
    """Recreate an *uninitialised* model purely from meta."""
    mtype = meta.get("model_type", "mlp")
    if mtype == "mlp":
        return PlugClassifier(int(meta["input_dim"])).to(device)
    if mtype == "cnn":
        return _make_model("cnn", input_dim=-1, meta=meta).to(device)
    raise ValueError(f"Unknown model_type {mtype!r}")


def load_artifacts(
    path: str,
    *,
    device: str | torch.device = "cpu",
) -> Tuple[torch.nn.Module, dict]:
    """
    Load weights + meta produced by :func:`save_artifacts`.
    *path* may be a prefix (no extension) or the weight file itself.
    """
    prefix = _path_prefix(path)

    # locate weights
    weights_path = None
    for ext in (".pt", ".pth"):
        cand = prefix + ext
        if os.path.isfile(cand):
            weights_path = cand
            break
    if weights_path is None:
        raise FileNotFoundError(f"No weight file (.pt or .pth) for prefix '{prefix}'.")

    # load or infer meta 
    meta_path = prefix + ".json"
    if os.path.isfile(meta_path):
        with open(meta_path) as fp:
            meta = json.load(fp)
    else:
        # fall back to minimal introspection for an MLP
        sd = torch.load(weights_path, map_location="cpu")
        lin_w = next((v for k, v in sd.items() if v.ndim == 2), None)
        if lin_w is None:
            raise RuntimeError("meta.json missing and could not infer architecture.")
        meta = {"model_type": "mlp", "input_dim": lin_w.shape[1]}
        LOGGER.warning("meta.json missing – inferred input_dim=%d", lin_w.shape[1])

    device = torch.device(device)
    model = _build_from_meta(meta, device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    LOGGER.info("Loaded model ← %s  (device=%s)", weights_path, device)
    return model, meta

def _read_features(src: Union[str, pd.DataFrame, np.ndarray]) -> Tuple[pd.DataFrame, dict | None]:
    """Return (DataFrame, meta_or_None) for a variety of *src* specs."""
    if isinstance(src, pd.DataFrame):
        return src, None
    if isinstance(src, np.ndarray):
        return pd.DataFrame(src), None
    if isinstance(src, str):
        if src.endswith(".npz"):
            return load_npz_features(src)
        return pd.read_csv(src, index_col=0), None
    raise TypeError(type(src))


# TRAINING – CROSS‑VALIDATION (BINARY CLASSIFICATION ONLY)
def cross_validate(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, pd.DataFrame, np.ndarray],
    *,
    model_type: str = "mlp",
    meta: dict | None = None,
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
    """
    Cross‑validation loop for binary classification.
    Returns per‑sample predicted probabilities. All artifacts
    saved under *out_dir/run_name*.
    """
    from datetime import datetime

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_out_dir = os.path.join(out_dir, run_name)
    os.makedirs(full_out_dir, exist_ok=True)

    # prepare labels
    if isinstance(y, pd.DataFrame):
        if label_col is None:
            raise ValueError("label_col required when y is a DataFrame.")
        y_vec = y[label_col].to_numpy()
    elif isinstance(y, pd.Series):
        y_vec = y.to_numpy()
    else:
        y_vec = np.asarray(y)

    # infer groups if requested 
    if groups is None and leave_out_col is not None:
        if isinstance(X, pd.DataFrame) and leave_out_col in X.columns:
            groups = X[leave_out_col]
        elif isinstance(y, pd.DataFrame) and leave_out_col in y.columns:
            groups = y[leave_out_col]
        else:
            raise KeyError(f"{leave_out_col!r} not found in X nor y.")
    if groups is not None:
        groups   = np.asarray(groups)
        splitter = LeaveOneGroupOut()
        split_gen = splitter.split(X, y_vec, groups)
        n_splits = len(np.unique(groups))
        mode     = f"LOGO ({n_splits} groups)"
    else:
        splitter  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=101)
        split_gen = splitter.split(X, y_vec)
        mode      = f"{n_splits}-fold stratified"

    LOGGER.info(
        "CV • mode=%s • model=%s • epochs=%d (eval_every=%d) • run=%s • out=%s",
        mode, model_type, n_epochs, eval_every, run_name, full_out_dir
    )

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    X_np, _ = _to_numpy(X)

    bce_loss = nn.BCEWithLogitsLoss()
    supcon   = SupConLoss().to(device)

    preds       = np.zeros(len(y_vec), np.float32)
    all_epochs  : list[list[int]]   = []
    all_metrics : list[list[float]] = []
    summary     : list[dict]        = []
    cv_start    = time.time()

    for fold, (tr, va) in enumerate(split_gen, 1):
        grp_name = np.unique(groups[va])[0] if groups is not None else "N/A"
        LOGGER.info("Fold %d/%d (group=%s) – training …", fold, n_splits, grp_name)

        model = _make_model(model_type, input_dim=X_np.shape[1], meta=meta).to(device)
        if model_type == "cnn":
            LOGGER.info(
                "CNN view: layers=%d parts=%d • contrastive_w=%.3f",
                len(meta["layers"]), len(meta["parts"]), contrastive_weight
            )

        opt = optim.Adam(model.parameters(), lr=learning_rate)
        ds  = torch.utils.data.TensorDataset(
            torch.tensor(X_np[tr]), torch.tensor(y_vec[tr], dtype=torch.float32)
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        epochs_checked: list[int]   = []
        metric_hist   : list[float] = []
        best_metric   = -float("inf")
        epochs_no_imp = 0

        for ep in range(1, n_epochs + 1):
            model.train()
            running_loss = 0.0

            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)

                # forward
                if model_type == "cnn":
                    logits, z = model(xb)
                    cls_loss  = bce_loss(logits, yb)
                    loss      = cls_loss + contrastive_weight * supcon(z, yb.long())
                else:
                    out    = model(xb)
                    logits = out.squeeze(1)
                    loss   = bce_loss(logits, yb)

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
                    p_tr, _  = _forward(
                        model,
                        torch.tensor(X_np[tr]).to(device),
                        model_type
                    )

                v = p_val.squeeze().cpu().numpy()
                t = p_tr.squeeze().cpu().numpy()

                val_auc = roc_auc_score(y_vec[va].astype(int), v)
                tr_auc  = roc_auc_score(y_vec[tr].astype(int), t)
                LOGGER.info(
                    "Fold %d ep %3d | train_auc %.4f val_auc %.4f",
                    fold, ep, tr_auc, val_auc
                )

                epochs_checked.append(ep)
                metric_hist.append(val_auc)
                p_val_final = v

                # early stopping
                if early_stopping:
                    if val_auc > best_metric:
                        best_metric   = val_auc
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
            "fold":        fold,
            "group":       str(grp_name),
            "final_auc":   metric_hist[-1],
            "epoch_final": epochs_checked[-1]
        })
        LOGGER.info(
            "Fold %d done – final_auc %.4f @ epoch %d",
            fold, metric_hist[-1], epochs_checked[-1]
        )

    # plot curves
    plt.figure()
    min_v = min(min(m) for m in all_metrics)
    for f, (eps, mets) in enumerate(zip(all_epochs, all_metrics), 1):
        plt.scatter(eps, mets, s=15, alpha=0.3)
        plt.plot(eps, pd.Series(mets).rolling(3, min_periods=1).mean(),
                 label=f"fold {f}")
    plt.ylabel("ROC‑AUC")
    plt.ylim(min_v - 0.05, 1.0)
    plt.title("Cross‑validation ROC‑AUC")
    plt.xlabel("epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(full_out_dir, "cv_curves.png")
    plt.savefig(png_path, dpi=120)
    plt.close()

    # overall metrics & summary JSON
    elapsed = time.time() - cv_start
    overall_auc = roc_auc_score(y_vec.astype(int), preds)
    LOGGER.info("CV done – overall ROC‑AUC %.4f | %.1fs", overall_auc, elapsed)
    summary_dict = {
        "overall_auc": overall_auc,
        "sec_total":   elapsed,
        "folds":       summary,
        "curves_png":  os.path.basename(png_path)
    }
    with open(os.path.join(full_out_dir, "cv_summary.json"), "w") as fp:
        json.dump(summary_dict, fp, indent=2)

    return preds


# 5.  FULL‑DATA FIT
def fit(
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    *,
    model_type: str = "mlp",
    meta: dict | None = None,
    n_epochs: int = 300,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda",
    contrastive_weight: float = 1.0,
    early_stopping: bool = False,
    patience: int = 10
):
    """
    Train on the full dataset and return the fitted model.
    Binary classification only.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    X_np, _ = _to_numpy(X)
    y_np    = np.asarray(y, dtype=np.float32)

    model     = _make_model(model_type, input_dim=X_np.shape[1], meta=meta).to(device)
    opt       = optim.Adam(model.parameters(), lr=learning_rate)
    supcon    = SupConLoss().to(device)
    bce_loss  = nn.BCEWithLogitsLoss()

    ds     = torch.utils.data.TensorDataset(
        torch.tensor(X_np), torch.tensor(y_np)
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    no_imp    = 0

    for ep in tqdm(range(n_epochs), desc="Full‑fit", ncols=80):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)

            if model_type == "cnn":
                logits, z = model(xb)
                loss = bce_loss(logits, yb) + contrastive_weight * supcon(z, yb.long())
            else:
                out    = model(xb)
                logits = out.squeeze(1)
                loss   = bce_loss(logits, yb)

            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(ds)
        if early_stopping:
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_imp    = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    LOGGER.info(
                        "Early stopping (no BCE improvement for %d epochs)",
                        patience
                    )
                    break

    model.eval()
    return model



def predict(
    features: Union[str, pd.DataFrame, np.ndarray],
    *,
    model_path: str,
    output_csv: str = "predictions.csv",
    response_csv: str | None = None,
    response_col: str = "answer",
    threshold: float = 0.5,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Run a forward pass and (optionally) score against *response_csv*.
    Binary classification only; predictions are probabilities.
    """
    # features 
    X_df, _ = _read_features(features)
    ids     = X_df.index.to_numpy()
    X_np    = X_df.values.astype(np.float32)

    # model 
    model, meta = load_artifacts(model_path, device=device)
    model_type  = meta.get("model_type", "mlp")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        probs_tensor, _ = _forward(
            model,
            torch.tensor(X_np, device=device),
            model_type,
        )
    probs = probs_tensor.squeeze().cpu().numpy()

    # write predictions 
    pred_df = pd.DataFrame({"id": ids, "prediction": probs})
    pred_df.to_csv(output_csv, index=False)
    LOGGER.info("Predictions → %s", output_csv)

    # optional evaluation 
    if response_csv is not None:
        gt_df   = pd.read_csv(response_csv, index_col="id")
        merged  = gt_df[[response_col]].join(pred_df.set_index("id"), how="inner")
        if merged.empty:
            LOGGER.warning("No overlapping IDs with ground‑truth; metrics skipped.")
        else:
            y_true = merged[response_col].values.astype(float)
            y_pred = merged["prediction"].values
            y_bin  = (y_pred >= threshold).astype(int)
            metrics = {
                "accuracy":  accuracy_score(y_true, y_bin),
                "roc_auc":   roc_auc_score(y_true, y_pred),
                "precision": precision_score(y_true, y_bin),
                "recall":    recall_score(y_true, y_bin),
                "f1":        f1_score(y_true, y_bin),
            }
            mpath = Path(output_csv).with_suffix(".metrics.json")
            with open(mpath, "w") as fp:
                json.dump(metrics, fp, indent=2)
            LOGGER.info("Metrics → %s", mpath)

    return pred_df
