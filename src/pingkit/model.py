from __future__ import annotations

# standard library
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Tuple, Union

# third‑party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm

plt.switch_backend("Agg")  # headless‑safe

LOGGER = logging.getLogger(__name__)
__all__ = [
    "PingClassifier",
    "PingContrastiveCNN",
    "fit",
    "cross_validate",
    "save_artifacts",
    "load_artifacts",
    "predict",
    "load_npz_features",
]

# helpers

def _select_device(preferred: str | torch.device = "cuda") -> torch.device:
    """Return a usable device, falling back to CPU if CUDA is unavailable."""
    return torch.device(preferred if torch.cuda.is_available() else "cpu")


def _json_encode(obj):
    """Handle NumPy scalars / arrays when dumping JSON."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj)} cannot be JSON‑encoded")


# model definitions

class pingClassifier(nn.Module):
    """Width‑scaling MLP for tabular data."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        *,
        n_examples: int | None = None,
        target_ratio: float = 5.0,
        p_drop: float = 0.3,
        out_floor: int = 16,
        width_cap: int = 128,
    ):
        super().__init__()
        self.n_examples = n_examples
        self.target_ratio = target_ratio
        self.p_drop = p_drop
        self.width_cap = width_cap

        if n_examples is None:
            fc1_w = 32
        else:
            est_fc1 = int(target_ratio * n_examples / input_dim)
            fc1_w = int(np.clip(est_fc1, 16, width_cap))

        fc2_w = max(16, fc1_w // 2)
        out_w = max(out_floor, fc2_w // 4)

        def _block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(i, o),
                nn.BatchNorm1d(o),
                nn.ReLU(),
                nn.Dropout(p_drop),
            )

        self.fc1 = _block(input_dim, fc1_w)
        self.fc2 = _block(fc1_w, fc2_w)
        self.res = nn.Sequential(_block(fc2_w, fc2_w), nn.Linear(fc2_w, fc2_w))
        self.out = nn.Sequential(_block(fc2_w, out_w), nn.Linear(out_w, num_classes))

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        LOGGER.info(
            "pingClassifier • d=%d N=%s target=%.1f ⇒ fc1=%d params=%d (≈%.1f×N)",
            input_dim,
            n_examples,
            target_ratio,
            fc1_w,
            n_params,
            n_params / (n_examples or 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        x = x + self.res(x)
        return self.out(x)  # logits


class pingCNNEncoder(nn.Module):
    """Capacity‑aware CNN encoder for 1‑D or grid‑like inputs."""

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
        self.use_1d = n_parts == 1
        in_ch = n_layers

        base = 16 if n_examples is None else int(np.clip(np.sqrt(target_ratio * n_examples / 10), 8, width_cap))
        c1, c2 = base, base * 2
        self.proj_dim = max(32, base * proj_mult)

        self.layer_norm = nn.LayerNorm(hidden)

        if self.use_1d:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_ch, c1, 5, padding=2),
                nn.BatchNorm1d(c1),
                nn.ReLU(),
                nn.Conv1d(c1, c2, 3, padding=1),
                nn.BatchNorm1d(c2),
                nn.ReLU(),
                nn.Conv1d(c2, self.proj_dim, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(p_drop),
            )
        else:
            k_h = 3 if n_parts > 2 else n_parts
            self.backbone = nn.Sequential(
                nn.Conv2d(in_ch, c1, (k_h, 5), padding=(k_h // 2, 2)),
                nn.BatchNorm2d(c1),
                nn.ReLU(),
                nn.Conv2d(c1, c2, 1),
                nn.BatchNorm2d(c2),
                nn.ReLU(),
                nn.Conv2d(c2, self.proj_dim, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p_drop),
            )

        LOGGER.info("CNN encoder base=%d proj_dim=%d", base, self.proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        return self.backbone(x)


class pingContrastiveCNN(nn.Module):
    """CNN encoder + MLP head tuned for supervised contrastive training."""

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
        self.n_examples = n_examples
        self.target_ratio = target_ratio
        self.p_drop = p_drop
        self.width_cap = width_cap
        self.proj_mult = proj_mult

        self.n_parts = n_parts
        self.n_layers = n_layers
        self.hidden = hidden

        self.encoder = pingCNNEncoder(
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
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(proj_dim, max(32, proj_dim // 4)),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(max(32, proj_dim // 4), num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        LOGGER.info(
            "pingContrastiveCNN • classes=%d params=%d (≈%.1f×N)",
            num_classes,
            n_params,
            n_params / (n_examples or 1),
        )

    def forward(self, flat_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = flat_x.size(0)
        if self.n_parts == 1:
            x = flat_x.view(b, self.n_layers, self.hidden)
        else:
            x = flat_x.view(b, self.n_layers, self.n_parts, self.hidden)
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature
        self.eps = 1e-8

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        f = torch.nn.functional.normalize(feats, p=2, dim=1)
        sim = (f @ f.T) / self.tau
        mask = (labels[:, None] == labels[None, :]).float()
        sim = sim * (1 - torch.eye(mask.size(0), device=mask.device))
        exp = torch.exp(sim) * (1 - torch.eye(mask.size(0), device=mask.device))
        log_prob = sim - torch.log(exp.sum(1, keepdim=True) + self.eps)
        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)
        return -mean_log_prob.mean()


# I/O utilities

def load_npz_features(npz_path: str) -> Tuple[pd.DataFrame, dict]:
    with np.load(npz_path, allow_pickle=True) as npz:
        data = npz["data"]
        ids = npz["columns"]
        parts = list(npz["parts"])
        layers = list(npz["layers"])
        hidden = int(npz["hidden_size"][0])
    df = pd.DataFrame(data.T, index=ids)
    meta = {"parts": parts, "layers": layers, "hidden_size": hidden}
    return df, meta


def _to_numpy(X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float32), X.index.to_numpy()
    return X.astype(np.float32), np.arange(X.shape[0])


def _make_model(
    model_spec: Union[str, Callable],
    *,
    input_dim: int,
    meta: dict | None = None,
    num_classes: int = 2,
    n_examples: int | None = None,
    target_ratio: float = 5.0,
    p_drop: float = 0.3,
    **model_kwargs
) -> nn.Module:
    """
    Create a model from specification.
    
    Args:
        model_spec: Either a string for built-in models ("mlp", "cnn") or a callable factory
        input_dim: Input feature dimension
        meta: Metadata (required for CNN)
        num_classes: Number of output classes
        n_examples: Number of training examples
        target_ratio: Parameter scaling ratio
        p_drop: Dropout probability
        **model_kwargs: Additional arguments passed to custom model factories
        
    Returns:
        nn.Module: Instantiated model
    """
    if isinstance(model_spec, str):
        # Built-in models
        if model_spec == "mlp":
            return pingClassifier(
                input_dim,
                num_classes,
                n_examples=n_examples,
                target_ratio=target_ratio,
                p_drop=p_drop,
            )
        elif model_spec == "cnn":
            if meta is None:
                raise ValueError("meta required for CNN.")
            return pingContrastiveCNN(
                len(meta["parts"]),
                len(meta["layers"]),
                meta["hidden_size"],
                num_classes=num_classes,
                n_examples=n_examples,
                target_ratio=target_ratio,
                p_drop=p_drop,
            )
        else:
            raise ValueError(f"Unknown built-in model '{model_spec}'. Available: ['mlp', 'cnn']")
    
    elif callable(model_spec):
        # Custom model factory
        try:
            # Try with full signature first
            return model_spec(
                input_dim=input_dim,
                num_classes=num_classes,
                n_examples=n_examples,
                meta=meta,
                target_ratio=target_ratio,
                p_drop=p_drop,
                **model_kwargs
            )
        except TypeError:
            # Fallback for simpler signatures - just pass the essentials
            try:
                return model_spec(input_dim, num_classes, **model_kwargs)
            except TypeError:
                # Last resort - minimal signature
                return model_spec(input_dim, num_classes)
    
    else:
        raise TypeError(f"model_spec must be str or callable, got {type(model_spec)}")


def _should_use_contrastive_loss(model_spec: Union[str, Callable], model: nn.Module) -> bool:
    """Determine if the model should use contrastive loss."""
    if isinstance(model_spec, str):
        return model_spec == "cnn"
    # For custom models, check if it's a pingContrastiveCNN or returns tuples
    return isinstance(model, pingContrastiveCNN)


def _make_class_weight_tensor(
    class_weight: Union[str, Sequence[float], torch.Tensor, None],
    y_np: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor | None:

    if class_weight is None:
        return None

    if isinstance(class_weight, str):
        if class_weight not in {"balanced", "auto"}:
            raise ValueError(f"Unknown class_weight string {class_weight!r}")
        counts = np.bincount(y_np, minlength=num_classes).astype(float)
        if (counts == 0).any():
            raise ValueError(
                "At least one class has zero frequency; cannot compute balanced weights."
            )
        weights = counts.sum() / (counts * num_classes)  # mean(weight)=1
        return torch.tensor(weights, dtype=torch.float32, device=device)

    # sequence or tensor
    weights = torch.as_tensor(class_weight, dtype=torch.float32, device=device)
    if weights.numel() != num_classes:
        raise ValueError(
            f"class_weight length ({weights.numel()}) must match num_classes ({num_classes})"
        )
    return weights


def _forward(model: nn.Module, xb: torch.Tensor, model_type: str) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Run model and normalise its output to (logits, embedding_opt)."""
    out = model(xb)
    if isinstance(out, tuple):
        logits, *extras = out
        z = extras[0] if extras else None
    else:
        logits = out
        z = None
    return logits, z


def _path_prefix(path: str) -> str:
    root, ext = os.path.splitext(path)
    return root if ext.lower() in {".pt", ".pth", ".bin"} else path


def save_artifacts(
    model: torch.nn.Module,
    *,
    path: str = "artifacts/ping",
    meta: dict | None = None,
    model_factory: Callable | None = None,  # NEW: reconstruction function
    model_kwargs: dict | None = None,       # NEW: factory arguments
) -> Tuple[str, str]:
    """Serialise weights (.pt) + mini‑meta (.json).
    
    For custom models, provide model_factory and model_kwargs for reconstruction:
    
    save_artifacts(
        model, 
        path="my_model",
        model_factory=my_probe_factory,
        model_kwargs={"hidden_dim": 128, "num_heads": 4}
    )
    """
    prefix = _path_prefix(path)
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    if isinstance(model, pingClassifier):
        mtype = "mlp"
        num_classes = model.out[-1].out_features
        meta_out: dict = {"model_type": mtype, "num_classes": num_classes}
        meta_out["input_dim"] = int(model.fc1[0].in_features)
        meta_out["n_examples"] = getattr(model, "n_examples", None)
        meta_out["target_ratio"] = getattr(model, "target_ratio", 5.0)
        meta_out["p_drop"] = getattr(model, "p_drop", 0.3)
        meta_out["width_cap"] = getattr(model, "width_cap", 128)
        
    elif isinstance(model, pingContrastiveCNN):
        mtype = "cnn"
        num_classes = model.classifier[-1].out_features
        meta_out: dict = {"model_type": mtype, "num_classes": num_classes}
        meta_out["parts"] = model.n_parts
        meta_out["layers"] = model.n_layers
        meta_out["hidden_size"] = model.hidden
        meta_out["n_examples"] = getattr(model, "n_examples", None)
        meta_out["target_ratio"] = getattr(model, "target_ratio", 5.0)
        meta_out["p_drop"] = getattr(model, "p_drop", 0.3)
        meta_out["width_cap"] = getattr(model, "width_cap", 64)
        meta_out["proj_mult"] = getattr(model, "proj_mult", 2)
        
    else:
        # Custom model handling
        mtype = "custom"
        
        # Try to infer num_classes from final layer
        num_classes = None
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Linear):
                num_classes = module.out_features
                break
                
        if num_classes is None:
            raise ValueError(
                "Could not infer num_classes from custom model. "
                "Ensure your model has a final nn.Linear layer, or provide num_classes in meta."
            )
            
        meta_out: dict = {
            "model_type": mtype, 
            "num_classes": num_classes,
            "custom_model_class": type(model).__name__,
        }
        
        # Try to infer input_dim from first layer
        input_dim = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                input_dim = module.in_features
                break
                
        if input_dim is not None:
            meta_out["input_dim"] = input_dim
        
        # Handle reconstruction info
        if model_factory is not None:
            import inspect
            try:
                # Try to save factory function info
                meta_out["factory_name"] = model_factory.__name__
                meta_out["factory_module"] = getattr(model_factory, '__module__', None)
                meta_out["factory_kwargs"] = model_kwargs or {}
                
                # Try to save source if it's a local function
                try:
                    source = inspect.getsource(model_factory)
                    meta_out["factory_source"] = source
                except (OSError, TypeError):
                    # Function source not available (e.g., built-in, compiled, etc.)
                    pass
                    
            except Exception as e:
                warnings.warn(f"Could not serialize model factory: {e}")
        else:
            warnings.warn(
                "Custom model saved without reconstruction info. "
                "Provide model_factory and model_kwargs to enable loading. "
                "Loading this model will require manual reconstruction.",
                UserWarning
            )

    # Apply user metadata
    if meta:
        meta_out.update(meta)

    weights_path = prefix + ".pt"
    torch.save(model.state_dict(), weights_path)

    meta_path = prefix + ".json"
    with open(meta_path, "w") as fp:
        json.dump(meta_out, fp, indent=2, default=_json_encode)

    LOGGER.info("Artifacts saved → %s (+ meta %s)", weights_path, meta_path)
    return os.path.abspath(weights_path), os.path.abspath(meta_path)


def _build_from_meta(meta: dict, device: Union[str, torch.device] = "cpu") -> torch.nn.Module:
    """Rebuild a model skeleton identical to the one that produced `meta`."""
    num_classes = int(meta.get("num_classes", 2))
    common_kwargs = dict(
        n_examples=meta.get("n_examples"),
        target_ratio=meta.get("target_ratio", 5.0),
        p_drop=meta.get("p_drop", 0.3),
        width_cap=meta.get("width_cap", 64),
    )

    mtype = meta.get("model_type", "mlp")
    if mtype == "mlp":
        model = pingClassifier(int(meta["input_dim"]), num_classes, **common_kwargs)
    elif mtype == "cnn":
        model = pingContrastiveCNN(
            len(meta["parts"]),
            len(meta["layers"]),
            meta["hidden_size"],
            num_classes=num_classes,
            proj_mult=meta.get("proj_mult", 2),
            **common_kwargs,
        )
    elif mtype == "custom":
        # Handle custom model reconstruction
        if "factory_source" in meta:
            # Try to reconstruct from saved source code
            try:
                factory_source = meta["factory_source"]
                factory_kwargs = meta.get("factory_kwargs", {})
                
                # Execute the source code in a clean namespace
                namespace = {"nn": nn, "torch": torch, "F": torch.nn.functional}
                exec(factory_source, namespace)
                
                # Get the factory function
                factory_name = meta.get("factory_name", "model_factory")
                if factory_name not in namespace:
                    raise ValueError(f"Factory function '{factory_name}' not found in source")
                    
                factory_fn = namespace[factory_name]
                
                # Try to call with various signatures
                try:
                    # First try with all standard parameters
                    model = factory_fn(
                        input_dim=meta.get("input_dim"),
                        num_classes=num_classes,
                        **factory_kwargs
                    )
                except TypeError:
                    # Fallback to minimal signature
                    input_dim = meta.get("input_dim")
                    if input_dim is None:
                        raise ValueError(
                            "input_dim not found in metadata. Cannot reconstruct custom model."
                        )
                    model = factory_fn(input_dim, num_classes, **factory_kwargs)
                    
            except Exception as e:
                raise ValueError(
                    f"Failed to reconstruct custom model from source: {e}\n"
                    f"You may need to manually reconstruct this model."
                )
        else:
            raise ValueError(
                f"Cannot reconstruct custom model '{meta.get('custom_model_class', 'Unknown')}'. "
                f"No reconstruction info saved. You need to manually create the model."
            )
    else:
        raise ValueError(f"Unknown model_type {mtype!r}")

    return model.to(device)


def load_artifacts(path: str, *, device: str | torch.device = "cpu") -> Tuple[torch.nn.Module, dict]:
    prefix = _path_prefix(path)

    weights_path = next((prefix + ext for ext in (".pt", ".pth") if os.path.isfile(prefix + ext)), None)
    if weights_path is None:
        raise FileNotFoundError(f"No weight file (.pt or .pth) for prefix '{prefix}'.")

    meta_path = prefix + ".json"
    if not os.path.isfile(meta_path):
        raise FileNotFoundError("meta.json missing; cannot rebuild model.")
    with open(meta_path) as fp:
        meta = json.load(fp)

    device = _select_device(device)
    model = _build_from_meta(meta, device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    LOGGER.info("Loaded model ← %s (device=%s)", weights_path, device)
    return model, meta


def _read_features(src: Union[str, pd.DataFrame, np.ndarray]) -> Tuple[pd.DataFrame, dict | None]:
    if isinstance(src, pd.DataFrame):
        return src, None
    if isinstance(src, np.ndarray):
        return pd.DataFrame(src), None
    if isinstance(src, str):
        if src.endswith(".npz"):
            return load_npz_features(src)
        return pd.read_csv(src, index_col=0), None
    raise TypeError(type(src))


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _macro_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_prob.ndim == 1 or y_prob.shape[1] == 1:
        return roc_auc_score(y_true, y_prob.ravel())
    elif y_prob.shape[1] == 2:
        # Binary classification: use probability of positive class
        return roc_auc_score(y_true, y_prob[:, 1])
    else:
        # Multiclass classification
        return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")


_METRIC_REGISTRY: dict[str, Tuple[MetricFn, str]] = {
    "roc_auc": (_macro_roc_auc, "max"),
    "accuracy": (lambda y, p: accuracy_score(y, p.argmax(1)), "max"),
    "macro_f1": (lambda y, p: f1_score(y, p.argmax(1), average="macro"), "max"),
}


def _ensure_metric(metric: str | MetricFn) -> Tuple[MetricFn | None, str]:
    """
    Return (metric_fn | None, mode).

    * If *metric* is a callable, we assume "higher‑is‑better".
    * If *metric* == "loss", we signal that we want to **minimise** the
      cross‑entropy returned by `_evaluate()` by returning (None, "min").
    * Otherwise look the metric up in the registry (higher‑is‑better).
    """
    if callable(metric):
        return metric, "max"          # user‑supplied scorer → maximise
    if metric == "loss":
        return None, "min"            # use CE/NLL, lower is better
    if metric not in _METRIC_REGISTRY:
        raise ValueError(f"Unknown metric {metric!r}")
    return _METRIC_REGISTRY[metric]


@dataclass
class EarlyStopping:
    """Stop training when the monitored metric has not improved after `patience` epochs."""
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "max"

    best: float | None = None
    bad_epochs: int = 0

    def step(self, score: float) -> bool:
        if self.best is None:
            self.best = score
            return False
        improve = (score - self.best) if self.mode == "max" else (self.best - score)
        if improve > self.min_delta:
            self.best = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs > self.patience


def _train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    model_type: str,
    opt: torch.optim.Optimizer,
    ce_loss: nn.CrossEntropyLoss,
    supcon: SupConLoss | None,
    contrastive_weight: float,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
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
        running += loss.item() * xb.size(0)

    return running / len(loader.dataset)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    model_type: str,
    metric_fn: MetricFn | None,
    ce_loss: nn.CrossEntropyLoss | None,
    device: torch.device,
    batch_size: int = 4096,
) -> Tuple[np.ndarray, float, float | None]:
    was_training = model.training
    model.eval()

    n = len(X)
    prob_chunks: list[torch.Tensor] = []
    running_ce = 0.0

    try:
        for i in range(0, n, batch_size):
            xb = torch.tensor(X[i : i + batch_size], device=device)
            yb = torch.tensor(y_true[i : i + batch_size], device=device)

            logits_b, _ = _forward(model, xb, model_type)
            probs_b = torch.softmax(logits_b, dim=1)
            prob_chunks.append(probs_b.cpu())

            if ce_loss is not None:
                running_ce += ce_loss(logits_b, yb).item() * len(yb)

        probs = torch.cat(prob_chunks, 0).numpy()
        score = math.nan if metric_fn is None else metric_fn(y_true, probs)
        loss_val = None if ce_loss is None else running_ce / n
        return probs, score, loss_val
    finally:
        if was_training:
            model.train()


def _make_validation_split(
    y: np.ndarray, *, val_fraction: float, random_state: int | None = 101
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=random_state
    )
    tr_idx, va_idx = next(splitter.split(np.zeros_like(y), y))
    return tr_idx, va_idx


# fit

# =====================================================================
# UPDATED fit()
# =====================================================================
def fit(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    *,
    model: Union[str, Callable] = "mlp",
    model_type: str | None = None,          # deprecated – kept for BC
    meta: dict | None = None,
    num_classes: int = 2,
    n_epochs: int = 300,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    device: str | torch.device = "cuda",
    contrastive_weight: float = 1.0,
    validation_data: Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray] | None = None,
    val_split: float | None = None,
    eval_metric: str | MetricFn = "roc_auc",
    metric: str | MetricFn | None = None,   # deprecated alias
    early_stopping: bool = True,
    patience: int = 10,
    random_state: int | None = 101,
    class_weight: Union[str, Sequence[float], torch.Tensor, None] = None,
    loss_fn: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "ce",
    **model_kwargs,
) -> Tuple[nn.Module, list[dict]]:

    if metric is not None:
        warnings.warn(
            "'metric' is deprecated; use 'eval_metric' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        eval_metric = metric

    # --------------- setup & data -------------------------------------
    device  = _select_device(device)
    X_np, _ = _to_numpy(X)
    y_np    = np.asarray(y, dtype=int)

    # --------------- backward compatibility ---------------------------
    if model_type is not None:
        warnings.warn(
            "The 'model_type' parameter is deprecated. Use 'model' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if model == "mlp":          # only override if using default
            model = model_type

    # --------------- validation handling ------------------------------
    if validation_data is not None:
        X_val_np, _ = _to_numpy(validation_data[0])
        y_val_np     = np.asarray(validation_data[1], dtype=int)
    elif val_split:
        tr_idx, va_idx = _make_validation_split(
            y_np, val_fraction=val_split, random_state=random_state
        )
        X_val_np, y_val_np = X_np[va_idx], y_np[va_idx]
        X_np,    y_np      = X_np[tr_idx], y_np[tr_idx]
    else:
        X_val_np = y_val_np = None
        early_stopping = False            # nothing to monitor

    # --------------- metric bootstrap ---------------------------------
    metric_fn, metric_mode = _ensure_metric(eval_metric)

    # --------------- class weighting ----------------------------------
    cw_tensor = _make_class_weight_tensor(
        class_weight, y_np, num_classes, device
    )

    # --------------- loss function ------------------------------------
    if callable(loss_fn):
        ce_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = loss_fn
    else:
        lf = loss_fn.lower()
        if lf == "ce":
            ce_loss = nn.CrossEntropyLoss(weight=cw_tensor)
        elif lf == "focal":
            # minimalist focal example; users can plug their own
            from torch.nn.functional import cross_entropy

            class _FocalLoss(nn.Module):
                def __init__(self, gamma: float = 2.0, weight=None):
                    super().__init__()
                    self.gamma = gamma
                    self.register_buffer("weight", weight)

                def forward(self, logits, target):
                    ce = cross_entropy(logits, target, weight=self.weight, reduction="none")
                    pt = torch.exp(-ce)
                    return ((1 - pt) ** self.gamma * ce).mean()

            ce_loss = _FocalLoss(gamma=2.0, weight=cw_tensor)
        else:
            raise ValueError(f"Unknown loss_fn {loss_fn!r}")

    # --------------- model / optimiser -------------------------------
    model_instance = _make_model(
        model,
        input_dim=X_np.shape[1],
        meta=meta,
        num_classes=num_classes,
        n_examples=len(X_np),
        **model_kwargs,
    ).to(device)

    opt     = optim.Adam(model_instance.parameters(), lr=learning_rate)
    supcon  = SupConLoss().to(device) if _should_use_contrastive_loss(model, model_instance) else None

    # --------------- dataloader ---------------------------------------
    ds     = torch.utils.data.TensorDataset(
        torch.tensor(X_np), torch.tensor(y_np, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    # --------------- training loop ------------------------------------
    stopper           = EarlyStopping(patience=patience, mode=metric_mode)
    history: list[dict] = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_metric                         = math.inf if metric_mode == "min" else -math.inf

    pbar = tqdm(range(1, n_epochs + 1), desc="Fit", ncols=80)
    for ep in pbar:
        tr_loss = _train_one_epoch(
            model_instance,
            loader,
            model_type=model if isinstance(model, str) else "custom",
            opt=opt,
            ce_loss=ce_loss,
            supcon=supcon,
            contrastive_weight=contrastive_weight,
            device=device,
        )

        # ---------- compute validation metric --------------------------
        val_metric = math.nan
        if X_val_np is not None:
            if eval_metric == "loss":
                _, _, val_metric = _evaluate(
                    model_instance,
                    X_val_np,
                    y_val_np,
                    model_type=model if isinstance(model, str) else "custom",
                    metric_fn=None,
                    ce_loss=ce_loss,
                    device=device,
                )
            else:
                _, val_metric, _ = _evaluate(
                    model_instance,
                    X_val_np,
                    y_val_np,
                    model_type=model if isinstance(model, str) else "custom",
                    metric_fn=metric_fn or _macro_roc_auc,
                    ce_loss=None,
                    device=device,
                )

        history.append({"epoch": ep, "train_loss": tr_loss, "val_metric": val_metric})
        pbar.set_postfix({"train_loss": f"{tr_loss:.4f}", "val_metric": f"{val_metric:.4f}"})

        # ---------- checkpointing -------------------------------------
        if X_val_np is not None and not math.isnan(val_metric):
            improved = (
                (metric_mode == "max" and val_metric > best_metric) or
                (metric_mode == "min" and val_metric < best_metric)
            )
            if improved:
                best_metric      = val_metric
                best_state_dict  = {k: v.detach().cpu().clone() for k, v in model_instance.state_dict().items()}

        # ---------- early‑stopping ------------------------------------
        if early_stopping and X_val_np is not None and stopper.step(val_metric):
            LOGGER.info("Early stopping at epoch %d (best=%.4f)", ep, stopper.best)
            break

    # --------------- restore best weights -----------------------------
    if best_state_dict is not None:
        model_instance.load_state_dict(best_state_dict)
        LOGGER.info("Restored best weights (val_metric %.4f).", best_metric)

    model_instance.eval()
    return model_instance, history


# cross‑validation

def cross_validate(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, pd.DataFrame, np.ndarray],
    *,
    model: Union[str, Callable] = "mlp",
    model_type: str | None = None,  # deprecated – kept for BC
    meta: dict | None = None,
    num_classes: int = 2,
    label_col: str | None = None,
    splitter: Iterable[Tuple[np.ndarray, np.ndarray]] | None = None,
    n_splits: int = 5,
    groups: Union[pd.Series, np.ndarray, None] = None,
    inner_val_split: float | None = 0.15,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    device: str | torch.device = "cuda",
    contrastive_weight: float = 1.0,
    eval_metric: str | MetricFn = "roc_auc",
    metric: str | MetricFn | None = None,   # deprecated alias
    early_stopping: bool = True,
    patience: int = 10,
    random_state: int | None = 101,
    eval_every: int = 1,
    out_dir: str = "artifacts",
    run_name: str | None = None,
    class_weight: Union[str, Sequence[float], torch.Tensor, None] = None,
    loss_fn: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "ce",
    # -----------------------------------------------------------------
    **model_kwargs,
) -> Tuple[np.ndarray, dict]:

    # -------- resolve deprecated alias --------------------------------
    if metric is not None:
        warnings.warn(
            "'metric' is deprecated; use 'eval_metric' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        eval_metric = metric

    # ---------- bookkeeping & splits ----------------------------------
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_out_dir = os.path.join(out_dir, run_name)
    os.makedirs(full_out_dir, exist_ok=True)

    # ---------- backward compatibility --------------------------------
    if model_type is not None:
        warnings.warn(
            "The 'model_type' parameter is deprecated. Use 'model' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if model == "mlp":
            model = model_type

    if isinstance(y, pd.DataFrame):
        if label_col is None:
            raise ValueError("label_col required when y is a DataFrame.")
        y_vec = y[label_col].to_numpy(dtype=int)
    elif isinstance(y, pd.Series):
        y_vec = y.to_numpy(dtype=int)
    else:
        y_vec = np.asarray(y, dtype=int)

    if splitter is None:
        if groups is not None:
            splitter = LeaveOneGroupOut().split(X, y_vec, groups)
            n_splits = len(np.unique(groups))
        else:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            ).split(X, y_vec)

    split_gen = list(splitter)

    # ---------- metric bootstrap --------------------------------------
    metric_fn, metric_mode = _ensure_metric(eval_metric)
    metric_name = (
        eval_metric
        if isinstance(eval_metric, str)
        else getattr(metric_fn, "__name__", "metric")
    )

    device  = _select_device(device)
    X_np, _ = _to_numpy(X)

    preds      = np.zeros((len(y_vec), num_classes), np.float32)
    curve_data = []
    fold_stats = []
    t0         = time.time()

    for fold, (tr, te) in enumerate(split_gen, 1):
        LOGGER.info("Fold %d/%d - training …", fold, len(split_gen))

        # ---------- inner validation split -----------------------------
        if inner_val_split and inner_val_split > 0.0:
            tr_idx, va_idx = _make_validation_split(
                y_vec[tr], val_fraction=inner_val_split, random_state=random_state
            )
            tr_inner, va_inner = tr[tr_idx], tr[va_idx]
        else:
            tr_inner, va_inner = tr, None

        fold_early_stop = early_stopping and (va_inner is not None)

        # ---------- dataloader ----------------------------------------
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X_np[tr_inner]),
            torch.tensor(y_vec[tr_inner], dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        # ---------- loss function (per‑fold weighting) -----------------
        device_fold = _select_device(device)
        ce_weight = _make_class_weight_tensor(
            class_weight, y_vec[tr_inner], num_classes, device_fold
        )

        if callable(loss_fn):
            ce_loss = loss_fn
        else:
            lf = loss_fn.lower()
            if lf == "ce":
                ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
            elif lf == "focal":
                from torch.nn.functional import cross_entropy

                class _FocalLoss(nn.Module):
                    def __init__(self, gamma: float = 2.0, weight=None):
                        super().__init__()
                        self.gamma = gamma
                        self.register_buffer("weight", weight)

                    def forward(self, logits, target):
                        ce = cross_entropy(
                            logits, target, weight=self.weight, reduction="none"
                        )
                        pt = torch.exp(-ce)
                        return ((1 - pt) ** self.gamma * ce).mean()

                ce_loss = _FocalLoss(gamma=2.0, weight=ce_weight)
            else:
                raise ValueError(f"Unknown loss_fn {loss_fn!r}")

        # ---------- model / opt ---------------------------------------
        model_instance = _make_model(
            model,
            input_dim=X_np.shape[1],
            meta=meta,
            num_classes=num_classes,
            n_examples=len(tr_inner),
            **model_kwargs,
        ).to(device_fold)

        opt     = optim.Adam(model_instance.parameters(), lr=learning_rate)
        stopper = EarlyStopping(patience=patience, mode=metric_mode)
        supcon  = SupConLoss().to(device_fold) if _should_use_contrastive_loss(model, model_instance) else None

        best_state_dict: dict[str, torch.Tensor] | None = None
        best_metric_fold                     = math.inf if metric_mode == "min" else -math.inf

        epochs_seen, metrics_seen = [], []

        # -------------- epoch loop ------------------------------------
        for ep in range(1, n_epochs + 1):
            _train_one_epoch(
                model_instance,
                loader,
                model_type=model if isinstance(model, str) else "custom",
                opt=opt,
                ce_loss=ce_loss,
                supcon=supcon,
                contrastive_weight=contrastive_weight,
                device=device_fold,
            )

            if ep % eval_every == 0 or ep == n_epochs:
                # ----- validation / train metric ----------------------
                val_metric = math.nan
                if va_inner is not None:
                    _, s_val, l_val = _evaluate(
                        model_instance,
                        X_np[va_inner],
                        y_vec[va_inner],
                        model_type=model if isinstance(model, str) else "custom",
                        metric_fn=None if eval_metric == "loss" else metric_fn,
                        ce_loss=ce_loss,
                        device=device_fold,
                    )
                    val_metric = l_val if eval_metric == "loss" else s_val

                _, s_tr, l_tr = _evaluate(
                    model_instance,
                    X_np[tr_inner],
                    y_vec[tr_inner],
                    model_type=model if isinstance(model, str) else "custom",
                    metric_fn=None if eval_metric == "loss" else metric_fn,
                    ce_loss=ce_loss,
                    device=device_fold,
                )
                train_metric = l_tr if eval_metric == "loss" else s_tr

                LOGGER.info(
                    "Fold %d ep %3d | train_%s %.4f val_%s %.4f",
                    fold, ep, metric_name, train_metric, metric_name, val_metric
                )

                epochs_seen.append(ep)
                metrics_seen.append(val_metric)

                # -------- best model for this fold --------------------
                if va_inner is not None and not math.isnan(val_metric):
                    improved = (
                        (metric_mode == "max" and val_metric > best_metric_fold) or
                        (metric_mode == "min" and val_metric < best_metric_fold)
                    )
                    if improved:
                        best_metric_fold = val_metric
                        best_state_dict  = {k: v.detach().cpu().clone() for k, v in model_instance.state_dict().items()}

                if fold_early_stop and stopper.step(val_metric):
                    LOGGER.info(
                        "Fold %d - early stop at epoch %d (best=%.4f)",
                        fold, ep, stopper.best
                    )
                    break

        # -------- restore best weights for this fold ------------------
        if best_state_dict is not None:
            model_instance.load_state_dict(best_state_dict)
            LOGGER.info("Fold %d - restored best weights (val %.4f)", fold, best_metric_fold)

        # -------- test‑split evaluation -------------------------------
        y_prob_test, s_test, l_test = _evaluate(
            model_instance,
            X_np[te],
            y_vec[te],
            model_type=model if isinstance(model, str) else "custom",
            metric_fn=None if eval_metric == "loss" else metric_fn,
            ce_loss=ce_loss,
            device=device_fold,
        )
        fold_metric_val = l_test if eval_metric == "loss" else s_test
        preds[te] = y_prob_test

        curve_data.append({"fold": fold, "epochs": epochs_seen, "metrics": metrics_seen})
        fold_stats.append({"fold": fold, "metric": fold_metric_val, "epochs_total": epochs_seen[-1]})

        LOGGER.info("Fold %d finished - %s %.4f", fold, metric_name, fold_metric_val)

    # ---------- aggregate results -------------------------------------
    if eval_metric == "loss":
        overall_metric = nn.CrossEntropyLoss()(
            torch.tensor(preds), torch.tensor(y_vec)
        ).item()
    else:
        overall_metric = metric_fn(y_vec, preds)

    elapsed = time.time() - t0
    cv_summary = {
        "overall_metric": overall_metric,
        "metric_name": metric_name,
        "metric_mode": metric_mode,
        "sec_total": elapsed,
        "folds": fold_stats,
    }

    # ---------- curve plotting  ----------------------------
    if curve_data and any(cd["metrics"] for cd in curve_data):
        plt.figure()
        vmin = min(min(cd["metrics"]) for cd in curve_data if cd["metrics"])
        for cd in curve_data:
            plt.scatter(cd["epochs"], cd["metrics"], s=12, alpha=0.4)
            plt.plot(
                cd["epochs"],
                pd.Series(cd["metrics"]).rolling(3, min_periods=1).mean(),
                label=f"fold {cd['fold']}",
            )
        plt.ylabel(metric_name)
        if metric_mode == "max":
            plt.ylim(vmin - 0.05, 1.0)
        plt.xlabel("epoch")
        plt.title("Cross‑validation learning curves")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        png_path = os.path.join(full_out_dir, "cv_curves.png")
        plt.savefig(png_path, dpi=120)
        plt.close()
        cv_summary["curves_png"] = os.path.basename(png_path)

    with open(os.path.join(full_out_dir, "cv_summary.json"), "w") as fp:
        json.dump(cv_summary, fp, indent=2)

    LOGGER.info("CV done - overall %.4f | %.1fs", overall_metric, elapsed)
    return preds, cv_summary


# prediction

def predict(
    features: Union[str, pd.DataFrame, np.ndarray],
    *,
    model_path: str,
    output_csv: str = "predictions.csv",
    response_csv: str | None = None,
    response_col: str = "answer",
    device: str = "cuda",
    batch_size: int = 4096,
    output_dir: str | None = None,
) -> pd.DataFrame:
    X_df, _ = _read_features(features)
    ids = X_df.index.to_numpy()
    X_np = X_df.values.astype(np.float32)

    model, meta = load_artifacts(model_path, device=device)
    model_type = meta.get("model_type", "mlp")
    num_classes = int(meta.get("num_classes", 2))

    device = _select_device(device)
    prob_chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.tensor(X_np[i : i + batch_size], device=device)
            logits_b, _ = _forward(model, xb, model_type)
            probs_b = torch.softmax(logits_b, dim=1)
            prob_chunks.append(probs_b.cpu())

    probs = torch.cat(prob_chunks, 0).numpy()

    if num_classes == 2:
        pred_df = pd.DataFrame({"id": ids, "prob_positive": probs[:, 1]}, columns=["id", "prob_positive"])
    else:
        pred_cols = {f"prob_class_{c}": probs[:, c] for c in range(num_classes)}
        pred_df = pd.DataFrame({"id": ids, **pred_cols})

    # Handle output directory
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_csv = os.path.join(output_dir, os.path.basename(output_csv))
    
    pred_df.to_csv(output_csv, index=False)
    LOGGER.info("Predictions → %s", output_csv)

    if response_csv is not None:
        gt_df = pd.read_csv(response_csv, index_col="id")
        merged = gt_df[[response_col]].join(pred_df.set_index("id"), how="inner")
        if merged.empty:
            LOGGER.warning("No overlapping IDs with ground truth; metrics skipped.")
        else:
            y_true = merged[response_col].values.astype(int)
            if num_classes == 2:
                y_pred_prob = merged["prob_positive"].values
                y_pred_cls = (y_pred_prob >= 0.5).astype(int)
                auc = roc_auc_score(y_true, y_pred_prob)
            else:
                prob_cols = [f"prob_class_{c}" for c in range(num_classes)]
                y_pred_prob = merged[prob_cols].values
                y_pred_cls = y_pred_prob.argmax(1)
                auc = roc_auc_score(y_true, y_pred_prob, multi_class="ovr", average="macro")

            metrics = {
                "accuracy": accuracy_score(y_true, y_pred_cls),
                "macro_f1": f1_score(y_true, y_pred_cls, average="macro"),
                "auc": auc,
                "precision_macro": precision_score(y_true, y_pred_cls, average="macro"),
                "recall_macro": recall_score(y_true, y_pred_cls, average="macro"),
            }
            mpath = Path(output_csv).with_suffix(".metrics.json")
            with open(mpath, "w") as fp:
                json.dump(metrics, fp, indent=2)
            LOGGER.info("Metrics → %s", mpath)

    return pred_df
