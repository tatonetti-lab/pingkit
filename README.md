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
    parts=["rs", "mlp"],
    eos_token="<eos>"
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

### Train and evaluate a model:

```python
from plug.model import load_npz_features, cross_validate

X, meta = load_npz_features("embeddings/features.npz")
y = pd.read_csv("data/labels.csv", index_col="id")

preds = cross_validate(
    X, y,
    model_type="mlp",
    task="classification",
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

## Documentation

Detailed documentation is coming soon. For now, please refer to the inline docstrings and examples provided in the source code.

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
