# Plug: Embedding Extraction and Modeling Utilities

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

Pattern Learning for Understanding Generation, or **Plug**, is a Python package that streamlines the journey from transformer activations to reproducible, capacity‑aware models. It includes utilities for

* extracting last‑token embeddings from any 🤗 HuggingFace `AutoModel*`,
* aggregating those embeddings into tidy feature matrices or compact `.npz` tensors, and
* training two neural architectures whose **size is chosen automatically from a simple data‑driven heuristic**.

---

## Features

| Area                       | What you get                                                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Embedding extraction**   | Fast batched capture of hidden states from decoder‑only and encoder‑decoder models, with EOS remapping and token masking.                               |
| **Aggregation helpers**    | Collapse many‑layer, many‑part tensors into row‑major matrices ready for Pandas/SciKit or keep them compressed in NumPy `.npz`.                         |
| **Capacity‑scaled models** | Residual MLP (`PlugClassifier`) and supervised‑contrastive CNN (`PlugContrastiveCNN`) that size themselves so that params ≈ `target_ratio × N_samples`. |
| **Training API**           | `fit` (single split) and `cross_validate` (outer + inner splits, early stopping, learning‑curve PNG).                                                   |
| **Artifact I/O**           | `save_artifacts` / `load_artifacts` store both weights **and** the meta needed to rebuild the exact geometry later.                                     |

---

## Installation

```bash
pip install git+https://github.com/tatonetti-lab/plug.git
```

> Instead of cloning, you can also add Plug as a sub‑module and install with `pip install -e .` if you plan to hack the source.

---

## Quick‑start examples

### 1 – Extract embeddings

```python
from plug.embeddings import embed_dataset

embed_dataset(
    data="data/prompts.csv",
    input_col="prompt",
    model_name="google/gemma-2-9b-it",
    output_dir="embeddings",
    device="cuda:0",
    layers=[0, 1, 7],
    parts=["attn", "mlp", "rs"],  # residual stream
    eos_token="<end_of_turn>"
)
```

### 2 – Aggregate to features

```python
from plug.extraction import extract_token_vectors

extract_token_vectors(
    embedding_dir="embeddings",
    parts=["rs", "mlp"],
    layers=[0, 1, 7],
    output_file="embeddings/features.npz",
    save_csv=True
)
```

### 3 – Cross‑validate evaluation

```python
import pandas as pd
from plug.model import load_npz_features, cross_validate

X, meta = load_npz_features("embeddings/features.npz")
y = pd.read_csv("data/labels.csv", index_col="id")

preds, summary = cross_validate(
    X, y,
    model_type="cnn",                # "mlp" or "cnn"
    meta=meta,                       # carries parts/layers/hidden info
    n_epochs=60,
    learning_rate=3e-4,
    batch_size=128,
    device="cuda",
    inner_val_split=0.1,
    early_stopping=True,
    patience=8,
)
print(summary["overall_metric"])
```

---

## Model architectures and sizing heuristics

### 1. Residual MLP — `PlugClassifier`

| Symbol | Meaning                                             |
| ------ | --------------------------------------------------- |
| `d`    | input feature dimension                             |
| `N`    | number of training examples (optional at inference) |
| `r`    | `target_ratio` (default 5.0)                        |
| `cap`  | `width_cap` (default 128)                           |
| `min`  | lower bound 16                                      |

The first hidden width is chosen so that the total parameter count tracks the data size:

$$
fc1_w = \mathrm{clip}\Big( \left\lfloor r \; N / d \right\rfloor ,\; \text{min},\; \text{cap} \Big)\tag{1}
$$

Subsequent widths follow simple halves/quarters:

```
fc2_w  = max(16, fc1_w // 2)
out_w  = max(out_floor, fc2_w // 4)   # out_floor = 16 by default
```

The network layout is

```
Input → FC1 → BN → ReLU → Dropout →
        FC2 → BN → ReLU → Dropout + Residual →
        Output block (BN, ReLU, Dropout, Linear)
```

> **Param heuristic**: By design the number of trainable parameters is $\approx r \times N$. That means you can scale to larger datasets simply by raising `N`—no manual width tuning required.

### 2. Supervised‑contrastive CNN — `PlugContrastiveCNN`

Let

* `parts` = number of embedding components (1 = just residual stream, 3 = rs+attn+mlp, …)
* `L` = number of transformer layers kept
* `h` = hidden size of the parent model

The encoder starts by computing a **base channel count**

$$
\text{base} = \mathrm{clip}\Big( \sqrt{\, r \; N / 10\,},\; 8,\; cap \Big)\tag{2}
$$

Channel widths are then

```
c1 = base
c2 = base * 2
proj_dim = max(32, base * proj_mult)   # proj_mult defaults to 2
```

The convolution stack (kernel height `k_h = 3` unless `parts ≤ 2`) is followed by global average pooling, flattening and dropout. The projection is fed to a small MLP head for logits **and** returned as an embedding for optional *supervised contrastive loss*:

$$
\mathcal L = \mathcal L_{\text{CE}} + \lambda \; \mathcal L_{\text{SupCon}}\tag{3}
$$

with `λ = contrastive_weight` (default 1.0) and temperature $\tau = 0.07$.

---

## Training utilities

* **`fit`** Single‑split training loop with mini‑batching, early stopping and history logging.
* **`cross_validate`** Outer CV with optional inner validation split, patience‑based early stopping, PNG learning curves and summary JSON.
* **Early‑stopping logic** Stops when the monitored metric has not improved by `min_delta` for `patience` consecutive epochs after the first plateau.

---

## Important notes for users

* **Inference without `N`** When you load a model only for prediction, `n_examples` may be `None`. The geometry is then replayed exactly as stored in the JSON meta.
* **Saved artifacts** `<prefix>.pt` (weights) + `<prefix>.json` (geometry + training hyper‑parameters). Keep them together.
* **Metrics** Default monitored metric is **macro ROC‑AUC** for classification; choose `"loss"` to monitor cross‑entropy instead.
* **Device selection** Every public API call accepts `device="cuda"` or `"cpu"`; if CUDA is unavailable the package silently falls back to CPU.
* **Reproducibility** `random_state` controls NumPy, PyTorch and scikit‑learn shuffling; set it for exact repeatability.

---

## Contributing

Pull requests and issues are welcome – please include a minimal reproducible example if reporting a bug.

---

## License

Plug is released under the MIT License. See the [LICENSE](LICENSE) file for full text.

---

## Contact

Jacob Berkowitz  ·  **[Jacob.Berkowitz2@cshs.org](mailto:Jacob.Berkowitz2@cshs.org)**
