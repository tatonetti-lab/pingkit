# Plug: Embedding Extraction and Modeling Utilities

Pattern Learning for Understanding Generation (Plug) is a Python package that turns transformer activations into reproducible, capacity‑aware models. It provides utilities for:

* Extracting hidden states and embeddings from any Hugging Face `AutoModel`.
* Aggregating those embeddings into feature matrices or compact `.npz` tensors.
* Training two neural architectures (MLP and CNN) that automatically size themselves based on data.

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/tatonetti-lab/plug.git
```

Alternatively, clone the repo and install in editable mode:

```bash
git clone https://github.com/tatonetti-lab/plug.git
cd plug
pip install -e .
```

---

## Function Reference

Below is a listing of all public functions and classes in each module, along with their parameters and behavior.

### `plug.embedding` Module

#### `load_model_and_tokenizer`

```python
def load_model_and_tokenizer(
    model_name: str = "google/gemma-2b-it",
    *,
    quantization: str | None = None,
    device_map: str | None = "auto",
) -> tuple[torch.nn.Module, transformers.PreTrainedTokenizer]
```

* **Description**: Loads a Hugging Face model (for hidden states) and its tokenizer. If `quantization` is specified (`"4bit"` or `"8bit"`), loads the model in quantized mode.
* **Parameters**:

  * `model_name` (str, default `"google/gemma-2b-it"`): Hugging Face model identifier.
  * `quantization` (str or None): If `"4bit"` or `"8bit"`, load the model in low‑bit mode; otherwise load full precision.
  * `device_map` (str or None): Device mapping strategy for loading (e.g., `"auto"` for automatic device placement).
* **Returns**: A tuple `(model, tokenizer)`, where `model` is a PyTorch `Module` with `output_hidden_states=True` and `tokenizer` is the corresponding HF tokenizer.

#### `embed_dataset`

```python
def embed_dataset(
    data: Union[pd.DataFrame, str, Iterable[str]],
    *,
    input_col: str | None = None,
    model_name: str = "google/gemma-2b-it",
    output_dir: str = "embeddings",
    layers: List[int] | None = None,
    parts: List[str] | None = None,
    pooling: Union[str, List[str]] = "last",
    eos_token: str | None = None,
    device: str | None = "auto",
    filter_non_text: bool = False,
)
```

* **Description**: Extracts token‑level embeddings for each row of `data`, applies pooling per layer and component (residual stream, attention, MLP), and saves CSV files under `output_dir/part/` for each row and layer.

* **Parameters**:

  * `data` (DataFrame, str, or iterable of str): If a DataFrame, must specify `input_col`. If a CSV path, loads it as DataFrame. If an iterable of strings, wraps into a DataFrame with column `__input__`.
  * `input_col` (str or None): Column name in `data` containing text inputs (required when `data` is a DataFrame).
  * `model_name` (str): Hugging Face model ID to use for embedding extraction.
  * `output_dir` (str): Root directory to write embedding CSVs; subdirectories are created per `part`.
  * `layers` (list of int or None): Indices of transformer layers to extract (default: all layers).
  * `parts` (list of str or None): Which sub‑components to save (`["rs", "attn", "mlp"]` by default).
  * `pooling` (str or list of str): Pooling strategy per token: one of `"first"`, `"last"`, `"mean"`, or `"max"`. Defaults to `"last"`.
  * `eos_token` (str or None): String to identify end‑of‑sequence tokens (if filtering).
  * `device` (str or None): Compute device (e.g., `"cpu"`, `"cuda:0"`, or `"auto"`).
  * `filter_non_text` (bool): If True, skip tokens that are punctuation/symbols, pandas duplicate suffixes, or contain `eos_token`.

* **Behavior**:

  1. Loads the model and tokenizer via `load_model_and_tokenizer`.
  2. Iterates over each input string:

     * Tokenizes with HF tokenizer.
     * Runs the model forward to collect hidden states and the outputs of attention & MLP sub‑modules via forward hooks.
     * Applies token filtering if `filter_non_text=True`.
     * For each `layer` in `layers`, obtains:

       * `seq_rs`: residual stream (hidden states) at that layer.
       * `seq_attn`: attention output from that block.
       * `seq_mlp`: MLP output from that block.
     * For each pooling `method` in `pooling`, computes a vector per `part` (`rs`, `attn`, `mlp`) by applying `_pooled` over valid token indices.
     * Writes each vector as a CSV of shape `(hidden_size, 1)` under `output_dir/<part>/<row_id>_L<layer>.csv`, where `row_id` is the DataFrame `id` column or `row_<idx>`.

#### `embed`

```python
def embed(
    inputs: Union[str, List[str]],
    *,
    model_name: str = "google/gemma-2b-it",
    layers: List[int] | None = None,
    parts: List[str] | None = None,
    pooling: Union[str, List[str]] = "last",
    eos_token: str | None = None,
    device: str | None = "auto",
    filter_non_text: bool = False,
) -> Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]]:
```

* **Description**: Returns embeddings in memory (no file I/O) for one or multiple input strings.
* **Parameters**:

  * `inputs` (str or list of str): Single string or list of strings to embed.
  * `model_name`, `layers`, `parts`, `pooling`, `eos_token`, `device`, `filter_non_text`: Same as `embed_dataset`.
* **Returns**: A nested dictionary:

  ```python
  {
    input_str: {
      layer_idx: {
        part: { "<token_key>": np.ndarray, ... },
        ...
      },
      ...
    },
    ...
  }
  ```

  * For each `input_str`, for each `layer`, for each `part` (`rs`, `attn`, `mlp`), a mapping from pooling key (token string for `"first"`/`"last"`, or pooling name for others) to a 1D NumPy array.

---

### `plug.extraction` Module

#### `extract_token_vectors`

```python
def extract_token_vectors(
    embedding_dir: str,
    *,
    parts: Union[str, Sequence[str]] = ("rs", "attn", "mlp"),
    layers: Union[int, Sequence[int], None] = None,
    output_file: Optional[str] = None,
    save_csv: bool = False,
    n_jobs: int = 8,
) -> str:
```

* **Description**: Scans the directory structure produced by `embed_dataset`, reads all per‑row, per‑layer CSVs, concatenates them into a single feature vector per `qid` (row), and saves a compressed `.npz` archive (and optionally a transposed CSV).
* **Parameters**:

  * `embedding_dir` (str): Root directory where `embed_dataset` created subfolders `rs/`, `attn/`, `mlp/` containing files named `<qid>_L<layer>.csv`.
  * `parts` (str or list of str): Which parts to include (`rs`, `attn`, `mlp`).
  * `layers` (int or list of int or None): If None, uses all discovered layers; else select specific layer indices.
  * `output_file` (str or None): Path (with or without `.npz`) to save results. Defaults to `embedding_dir/results/<parts>_L<layers>_stacked.npz`.
  * `save_csv` (bool): If True, also write a transposed CSV (`.csv`) alongside the `.npz`.
  * `n_jobs` (int): Number of parallel workers to use when reading and concatenating.
* **Behavior**:

  1. Discovers all `<qid>_L<layer>.csv` files for the first part.
  2. Infers number of layers, hidden size, and sample IDs (`qids`).
  3. Optionally restrict `layers` to provided indices.
  4. For each `qid`, concatenates all `parts` and `layers` in order into a single 1D array of length `hidden_size * len(parts) * len(layers)`.
  5. Constructs a Pandas DataFrame of shape `(feature_count, n_samples)` and saves:

     * A compressed `.npz` containing `data` (2D array: feature\_count × n\_samples), `columns` (sample IDs), `parts`, `layers`, `hidden_size`.
     * If `save_csv=True`, saves a transposed CSV (`n_samples × feature_count`).
* **Returns**: The path to the saved `.npz` file.

---

### `plug.model` Module

#### `load_npz_features`

```python
def load_npz_features(npz_path: str) -> Tuple[pd.DataFrame, dict]:
```

* **Description**: Loads a compressed `.npz` produced by `extract_token_vectors`. Returns a Pandas DataFrame with rows indexed by sample ID and columns as features, plus a metadata dictionary.
* **Parameters**:

  * `npz_path` (str): Path to the `.npz` file.
* **Returns**: `(df, meta)`:

  * `df` (DataFrame): shape `(n_samples, n_features)`, where `df.index` are sample IDs.
  * `meta` (dict): Contains keys `"parts"`, `"layers"`, and `"hidden_size"`.

#### `fit`

```python
from typing import Union, Tuple, List, dict

def fit(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    *,
    model_type: str = "mlp",
    meta: dict | None = None,
    num_classes: int = 2,
    n_epochs: int = 300,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    device: str | torch.device = "cuda",
    contrastive_weight: float = 1.0,
    validation_data: Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray] | None = None,
    val_split: float | None = None,
    metric: str | Callable[[np.ndarray, np.ndarray], float] = "roc_auc",
    early_stopping: bool = True,
    patience: int = 10,
    random_state: int | None = 101,
) -> Tuple[nn.Module, List[dict]]:
```

* **Description**: Trains a model (MLP or CNN) on features `X` and labels `y` using either a provided validation split or an internal `val_split`, with early stopping and training history logging.
* **Parameters**:

  * `X`: Feature matrix (DataFrame or NumPy) of shape `(n_samples, n_features)`.
  * `y`: Labels (Series or 1D array) of shape `(n_samples,)`.
  * `model_type`: `"mlp"` or `"cnn"`. If `"cnn"`, `meta` must be provided to reconstruct CNN input shape.
  * `meta`: Metadata dict (from `load_npz_features`) containing `"parts"`, `"layers"`, `"hidden_size"` (required if `model_type="cnn"`).
  * `num_classes`: Number of output classes.
  * `n_epochs`: Maximum number of epochs to train.
  * `learning_rate`: Optimizer learning rate.
  * `batch_size`: Mini‑batch size.
  * `device`: Compute device (`"cuda"` or `"cpu"`).
  * `contrastive_weight`: Weight λ for supervised contrastive loss (only used if `model_type="cnn"`).
  * `validation_data`: Tuple `(X_val, y_val)`. If provided, uses this as hold‑out validation set.
  * `val_split`: Fraction of data to set aside for validation (if `validation_data` is None).
  * `metric`: Either a registered metric name (`"roc_auc"`, `"accuracy"`, `"macro_f1"`) or a callable `metric(y_true, y_prob)`.
  * `early_stopping`: If True, enable early stopping on validation metric.
  * `patience`: Number of epochs with no improvement before stopping.
  * `random_state`: Seed for reproducibility (controls data shuffling).
* **Returns**: `(model, history)`:

  * `model`: Trained `nn.Module` (in eval mode).
  * `history`: List of dicts, each with keys `"epoch"`, `"train_loss"`, and `"val_metric"`.

#### `cross_validate`

```python
from typing import Union, Tuple, Any, List, Dict

def cross_validate(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, pd.DataFrame, np.ndarray],
    *,
    model_type: str = "mlp",
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
    metric: str | Callable[[np.ndarray, np.ndarray], float] = "roc_auc",
    early_stopping: bool = True,
    patience: int = 10,
    random_state: int | None = 101,
    eval_every: int = 1,
    out_dir: str = "artifacts",
    run_name: str | None = None,
) -> Tuple[np.ndarray, dict]:
```

* **Description**: Performs cross‑validation training and evaluation. Splits data into `n_splits` (using `StratifiedKFold` or `LeaveOneGroupOut` if `groups` provided), optionally with inner validation splits. Trains per fold, logs metrics and learning curves, saves artifacts under `out_dir/<run_name>/`.
* **Parameters**: Similar to `fit`, plus:

  * `label_col`: If `y` is a DataFrame, column name for labels.
  * `splitter`: Custom sequence of `(train_idx, test_idx)` splits.
  * `n_splits`: Number of outer folds (ignored if `splitter` provided).
  * `groups`: Group labels for leave‑one‑group‑out splitting.
  * `inner_val_split`: Fraction of each training fold to use for validation.
  * `eval_every`: Evaluate metrics every `eval_every` epochs.
  * `out_dir`: Parent directory to save fold‑level artifacts (JSON summary + learning curve PNG).
  * `run_name`: Subdirectory name for this run; defaults to timestamp.
* **Returns**: `(preds, cv_summary)`:

  * `preds`: NumPy array `(n_samples, num_classes)` of predicted probabilities.
  * `cv_summary`: Dict containing overall metric, per‑fold stats, elapsed time, and learning curve PNG path.

#### `save_artifacts`

```python
from typing import Tuple

def save_artifacts(
    model: torch.nn.Module,
    *,
    path: str = "artifacts/plug",
    meta: dict | None = None,
) -> Tuple[str, str]:
```

* **Description**: Saves a trained `PlugClassifier` or `PlugContrastiveCNN` to disk: weights (`.pt`) and metadata (`.json`). The metadata includes model geometry (input\_dim, parts, layers, hidden\_size), hyperparameters (`n_examples`, `target_ratio`, `p_drop`, `width_cap`, and `proj_mult` for CNN), plus any additional user‑supplied `meta`.
* **Parameters**:

  * `model`: Trained PyTorch model instance (`PlugClassifier` or `PlugContrastiveCNN`).
  * `path`: File prefix (without extension) for writing; `.pt` and `.json` are appended.
  * `meta`: Optional extra metadata to include in the JSON.
* **Returns**: Tuple of absolute paths `(weights_path, meta_path)`.

#### `load_artifacts`

```python
from typing import Tuple

def load_artifacts(
    path: str,
    *,
    device: str | torch.device = "cpu",
) -> Tuple[torch.nn.Module, dict]:
```

* **Description**: Loads saved weights (`.pt`) and metadata (`.json`) from `save_artifacts`. Reconstructs the model skeleton via `_build_from_meta`, loads weights, sets to `eval()` mode, and returns the model and meta.
* **Parameters**:

  * `path`: File prefix (with or without `.pt`/`.pth` extension).
  * `device`: Compute device (`"cpu"` or `"cuda"`).
* **Returns**: `(model, meta)`.

#### `predict`

```python
from typing import Union

def predict(
    features: Union[str, pd.DataFrame, np.ndarray],
    *,
    model_path: str,
    output_csv: str = "predictions.csv",
    response_csv: str | None = None,
    response_col: str = "answer",
    device: str = "cuda",
    batch_size: int = 4096,
) -> pd.DataFrame:
```

* **Description**: Loads features (from `.npz`, CSV, DataFrame, or NumPy array), loads a saved model via `load_artifacts`, runs inference to compute predicted probabilities, and writes an output CSV. If `response_csv` is provided, computes metrics against ground‑truth labels.
* **Parameters**:

  * `features`: Path to features (`.npz` or CSV), or in‑memory DataFrame/NumPy.
  * `model_path`: Path prefix to saved weights/JSON (as in `save_artifacts`).
  * `output_csv`: Path for writing predictions. For binary classification, writes columns `id` and `prob_positive`. For multiclass, writes `prob_class_{i}`.
  * `response_csv`: Optional CSV with ground truth, indexed by `id`, containing a column `response_col`.
  * `response_col`: Column name in `response_csv` of true labels.
  * `device`: `"cuda"` or `"cpu"` for inference.
  * `batch_size`: Batch size for inference.
* **Returns**: A Pandas DataFrame of predictions with `id` and `prob_*` columns.

#### `PlugClassifier` Class

```python
class PlugClassifier(nn.Module):
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
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

* **Description**: A capacity‑aware MLP for tabular features. The first hidden layer width is chosen so that the total parameter count ≈ `target_ratio × n_examples`, clipped between 16 and `width_cap`. Residual connections and dropout are used.
* **Parameters**:

  * `input_dim`: Number of input features (columns of `X`).
  * `num_classes`: Number of output classes.
  * `n_examples`: Number of training examples (influences layer sizing).
  * `target_ratio`: Desired ratio between total params and `n_examples`.
  * `p_drop`: Dropout probability.
  * `out_floor`: Minimum width of the output penultimate layer (defaults to 16).
  * `width_cap`: Maximum width for the first hidden layer.
* **Attributes**:

  * `.fc1`, `.fc2`, `.res`, `.out`: Sequential modules implementing the network.
* **`forward(x)`**:

  * Passes `x` through `fc1`, `fc2`, adds residual from `res(x)`, then through `out`.
  * Returns raw logits (no softmax).

#### `PlugContrastiveCNN` Class

```python
class PlugContrastiveCNN(nn.Module):
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
        ...
    def forward(self, flat_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
```

* **Description**: A supervised‑contrastive CNN encoder that consumes flattened features of shape `(batch_size, n_parts × n_layers × hidden)`. Internally reshapes to `(batch_size, n_layers, n_parts, hidden)` (or `[..., hidden]` if `n_parts==1`), normalizes, applies 1D/2D convolutions, global pooling, their output flows into a projection head that outputs both logits and embeddings for contrastive loss.
* **Parameters**:

  * `n_parts`: Number of embedding parts (e.g., 3 if using `rs`, `attn`, `mlp`).
  * `n_layers`: Number of transformer layers in features.
  * `hidden`: Hidden size of parent model (dimension per token embedding).
  * `num_classes`: Output classes.
  * `n_examples`, `target_ratio`, `width_cap`, `proj_mult`, `p_drop`: Same heuristics as `PlugClassifier`, applied to CNN channel widths.
* **Attributes**:

  * `.encoder`: `PlugCNNEncoder` instance.
  * `.classifier`: MLP head for final logits.
* **`forward(flat_x)`**:

  * Reshapes `flat_x` to `(batch, n_layers, n_parts, hidden)` (or `(batch, n_layers, hidden)` if `n_parts==1`).
  * Applies `encoder`, then `classifier`:

    * Returns `(logits, embedding)`.

#### `SupConLoss` Class

```python
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        ...
    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ...
```

* **Description**: Implements Supervised Contrastive Loss as in Khosla et al. (2020). Given a batch of normalized features `feats` and labels `labels`, computes pairwise cosine similarities, encourages same‑class pairs to be closer.
* **Parameters**:

  * `temperature`: Scaling factor for the similarity logits.
* **`forward(feats, labels)`**:

  * Returns a scalar contrastive loss.

---

## Walk‑through Example

Below is a complete example showing how to extract embeddings from text, aggregate them, load features, train a model with `fit`, and make predictions.

```python
#!/usr/bin/env python3
# Example: text → embeddings → features → training → prediction

import os
import pandas as pd

# 1. Extract embeddings from raw text
# -----------------------------------
# Assume you have a CSV `data/prompts.csv` with columns: id, prompt
# and a separate `data/labels.csv` with columns: id, label (0/1 or multiclass).

from plug.embedding import embed_dataset

# Create an output directory for embeddings
os.makedirs("embeddings", exist_ok=True)

embed_dataset(
    data="data/prompts.csv",       # Path to CSV with columns 'id' and 'prompt'
    input_col="prompt",            # Column containing text inputs
    model_name="google/gemma-2b-it",  # HF model to use
    output_dir="embeddings",       # Root dir for per‑part subfolders
    device="cuda:0",               # Or "cpu"
    layers=[0, 1, 7],               # Extract layers 0, 1, and 7
    parts=["rs", "attn", "mlp"],
    pooling=["mean"],              # Pool by mean over valid tokens
    eos_token="<end_of_turn>",     # Token string to identify EOS in filtering
    filter_non_text=True,            # Skip punctuation/duplicates
)

# This will write files like:
#   embeddings/rs/<id>_L00.csv
#   embeddings/attn/<id>_L00.csv
#   embeddings/mlp/<id>_L00.csv
#   embeddings/rs/<id>_L01.csv, etc.


# 2. Aggregate embeddings into feature matrices
# ---------------------------------------------

from plug.extraction import extract_token_vectors

# Collapse the CSVs into a compressed NPZ and optional CSV
npz_path = extract_token_vectors(
    embedding_dir="embeddings",   # Directory written by embed_dataset
    parts=["rs", "attn", "mlp"],
    layers=[0, 1, 7],              # Same layers as above
    output_file="embeddings/features",  # Will append .npz
    save_csv=True,                 # Also write a transposed CSV
)
# npz_path is now "embeddings/features.npz"

# You can inspect the CSV if desired:
#   embeddings/features.csv  (rows indexed by id, columns as features)


# 3. Load features and labels for training
# -----------------------------------------

from plug.model import load_npz_features, fit, save_artifacts, predict

# Load the NPZ into a DataFrame and metadata
X_df, meta = load_npz_features(npz_path)
# X_df: DataFrame shape (n_samples, n_features), index is id

# Load labels CSV; assume it has columns ['id', 'label']
labels_df = pd.read_csv("data/labels.csv", index_col="id")
# Align features and labels
common_ids = X_df.index.intersection(labels_df.index)
X_train = X_df.loc[common_ids]
y_train = labels_df.loc[common_ids, "label"].astype(int)

# 4. Train a model with `fit`
# ----------------------------

# Create a directory to store trained artifacts
os.makedirs("artifacts", exist_ok=True)

model, history = fit(
    X_train,
    y_train,
    model_type="cnn",       # "mlp" or "cnn" (requires meta for "cnn")
    meta=meta,               # Necessary when model_type="cnn"
    num_classes=len(y_train.unique()),
    n_epochs=100,
    learning_rate=3e-4,
    batch_size=64,
    device="cuda:0",        # Or "cpu"
    contrastive_weight=1.0,  # Only used if model_type="cnn"
    val_split=0.1,           # Reserve 10% for validation internally
    metric="roc_auc",       # Monitor ROC‑AUC
    early_stopping=True,
    patience=10,
    random_state=42,
)

# `history` is a list of dicts: [{'epoch':1, 'train_loss':..., 'val_metric':...}, ...]
for record in history[-5:]:  # print last 5 epochs
    print(record)

# 5. Save model weights + metadata
# --------------------------------

weights_path, meta_path = save_artifacts(
    model,
    path="artifacts/plug_final",  # Will create plug_final.pt + plug_final.json
    meta={"created_by": "example_script"},
)
print("Saved model to:", weights_path)
print("Saved meta to:", meta_path)

# 6. Make predictions on new data
# -------------------------------

# Suppose you have a set of test prompts:
# test_prompts.csv with columns ['id', 'prompt']
embed_dataset(
    data="data/test_prompts.csv",
    input_col="prompt",
    model_name="google/gemma-2b-it",
    output_dir="embeddings_test",
    device="cuda:0",
    layers=[0, 1, 7],
    parts=["rs", "attn", "mlp"],
    pooling=["mean"],
    eos_token="<end_of_turn>",
    filter_non_text=True,
)

npz_test = extract_token_vectors(
    embedding_dir="embeddings_test",  
    parts=["rs", "attn", "mlp"],
    layers=[0, 1, 7],
    output_file="embeddings_test/features_test",
    save_csv=False,
)

# Use `predict` to load features and run inference
pred_df = predict(
    features=npz_test,
    model_path="artifacts/plug_final",  # Prefix to .pt/.json
    output_csv="test_predictions.csv",
    response_csv=None,   # If no ground truth available
    device="cuda:0",
    batch_size=256,
)

print(pred_df.head())  # Contains 'id' and 'prob_positive' (binary) or 'prob_class_{i}'
```

---

## License

Plug is released under the MIT License. See the [LICENSE](LICENSE) file for details.


## Contact

Jacob Berkowitz  ·  **[Jacob.Berkowitz2@cshs.org](mailto:Jacob.Berkowitz2@cshs.org)**
