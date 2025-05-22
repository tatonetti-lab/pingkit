# Plug: Embedding Extraction and Modeling Utilities

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

Pattern Learning for Understanding Generation, or **Plug**, is a Python package designed to simplify modeling using transformer-based embeddings. It provides optimized utilities for capturing last-token activations from HuggingFace AutoModels, aggregating embeddings into structured datasets, and training robust neural network models.

---

## Features

- **Efficient Embedding Extraction**: Capture last-token embeddings from transformer models (e.g., Gemma, GPT-2, etc.).
- **Flexible Aggregation**: Easily aggregate embeddings across layers and model components (attention, MLP, residual streams).
- **Robust Modeling**: Train and evaluate neural network models (MLP and CNN architectures) with built-in cross-validation and metrics.
- **Easy-to-use API**: Simple and intuitive functions for embedding extraction, aggregation, and modeling.

---

## Installation

Install from source:

```bash
git clone https://github.com/tatonetti-lab/plug.git
cd plug
pip install .
```

---

## Quickstart

### Extract embeddings from a HuggingFace model:

```python
from plug.embeddings import embed_dataset

embed_dataset(
    data="data/prompts.csv",
    input_col="prompt",
    model_name="google/gemma-2-9b-it",
    output_dir="embeddings",
    device="cuda:0",
    layers=[0, 1, 7],
    parts=["attn", "mlp", "rs"],
    eos_token="<end_of_turn>"
)
```

### Aggregate embeddings into a structured dataset:

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

### Evaluate a model:

```python
from plug.model import load_npz_features, cross_validate

X, meta = load_npz_features("embeddings/features.npz")
y = pd.read_csv("data/labels.csv", index_col="id")

preds = cross_validate(
    X, y,
    model_type="mlp",
    meta=meta,
    n_epochs=50,
    learning_rate=1e-3,
    batch_size=64,
    device="cuda",
    early_stopping=True,
    patience=5
)
```

---

### Model Architectures

Plug provides two primary neural network architectures for modeling embeddings: a **Residual MLP** (`PlugClassifier`) and a **CNN-based model** (`PlugContrastiveCNN`). Both models are designed to be robust, interpretable, and easy to configure.

---

### 1. Residual MLP (`PlugClassifier`)

The Residual MLP is a fully-connected neural network with a residual connection, designed to adapt its width based on the input embedding dimension (`input_dim`). The architecture consists of:

- **Input Layer**: Accepts embedding vectors of dimension `input_dim`.
- **Hidden Layers**:
  - **First hidden layer**: Width is calculated as `fc1_w = clip(4 × input_dim, min=128, max=1024)`.
  - **Second hidden layer**: Width is half of the first layer (`fc2_w = fc1_w // 2`).
  - **Residual connection**: A parallel residual block with the same width as the second hidden layer (`fc2_w`), added to the output of the second hidden layer.
- **Output Layer**: A final linear layer reduces the dimension to a single scalar output (logit). For classification tasks, a sigmoid activation is applied externally.

**Parameter Calculation**:

- `fc1_w = clip(4 × input_dim, min=128, max=1024)`
- `fc2_w = fc1_w // 2`
- `out_w = max(64, fc2_w // 4)`

**Total Layers**: 3 fully-connected layers plus one residual block.

---

### 2. CNN-based Model (`PlugContrastiveCNN`)

The CNN-based model leverages convolutional layers to capture spatial relationships across embedding layers and parts (e.g., residual stream, attention, MLP). It consists of two main components:

- **CNN Encoder (`PlugCNNEncoder`)**:
  - **Input Shape**: `(batch_size, n_layers, hidden_size)` if using a single embedding part, or `(batch_size, n_layers, n_parts, hidden_size)` if multiple parts are used.
  - **Normalization**: Layer normalization applied across the embedding dimension.
  - **Convolutional Backbone**:
    - For single-part embeddings (`n_parts=1`), uses 1D convolutions:
      - Conv1d → BatchNorm → ReLU → Conv1d → BatchNorm → ReLU → Conv1d → ReLU → AdaptiveAvgPool1d → Flatten → Dropout
    - For multi-part embeddings (`n_parts>1`), uses 2D convolutions:
      - Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → Conv2d → ReLU → AdaptiveAvgPool2d → Flatten → Dropout
  - **Projection Dimension**: Final embedding dimension (`proj_dim`) defaults to 128.

- **Classifier Head**:
  - A small fully-connected network maps the CNN encoder output (`proj_dim`) to a single scalar logit.
  - Structure: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear (output).

**Parameter Calculation**:

- CNN channel sizes are dynamically set based on the number of layers (`n_layers`):
  - `c1 = max(128, 4 × n_layers)`
  - `c2 = max(256, 2 × c1)`
- Kernel sizes:
  - Single-part (1D): kernels of size 5 and 3.
  - Multi-part (2D): kernels of height `k_h = min(3, n_parts)` and width 5.

**Total Layers**: 3 convolutional layers in the encoder, followed by 3 linear layers in the classifier head.

---

### Early Stopping

Both models support early stopping during training to prevent overfitting:

- **Mechanism**: After each evaluation epoch, the validation metric (ROC-AUC for classification, MSE for regression) is monitored.
- **Patience**: If the validation metric does not improve for a specified number of epochs (`patience`, default=10), training stops early.
- **Best Model Selection**: The model state corresponding to the best validation metric is retained.

---

### Contrastive Loss (CNN only)

The CNN model optionally incorporates a supervised contrastive loss (`SupConLoss`) to improve embedding quality:

- **Purpose**: Encourages embeddings of samples from the same class to be closer together, and embeddings from different classes to be farther apart.
- **Temperature Parameter**: Default temperature (`τ`) is set to 0.07.
- **Weighting**: The contrastive loss is combined with the standard classification loss (BCE) using a configurable weight (`contrastive_weight`, default=1.0).

---

### Important Notes

- **Activation Functions**: Both models use ReLU activations internally. For classification tasks, the final sigmoid activation is applied externally during inference.
- **Device Management**: Models automatically detect and utilize GPU (`cuda`) if available, otherwise defaulting to CPU.
- **Batch Size and Learning Rate**: Default values (`batch_size=128`, `learning_rate=1e-3`) are provided, but tuning these hyperparameters based on your dataset is recommended.

---


## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📧 Contact

- **Author**: Jacob Berkowitz
- **Email**: Jacob.Berkowitz2@cshs.org
