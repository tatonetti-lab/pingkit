# Pingkit: Embedding Extraction and Modeling Utilities

**P**robing **IN**ternal states of **G**enerative models Kit (pingkit) trains reproducible, capacity‑aware ping models from transformer activations. It provides utilities for:

* Extracting hidden states and embeddings from any Hugging Face `AutoModel`.
* Aggregating those embeddings into feature matrices or compact `.npz` tensors.
* Training two neural architectures (MLP and CNN) that automatically size themselves based on data.
* Creating custom probes and models tailored to your specific research needs.

---

## Installation

Install most stable version:

```bash
pip install pingkit
```

Install latest dev version from GitHub:

```bash
pip install git+https://github.com/tatonetti-lab/pingkit.git
```

Alternatively, clone the repo and install in editable mode:

```bash
git clone https://github.com/tatonetti-lab/pingkit.git
cd ping
pip install -e .
```

---

## Tutorials

For advanced usage including creating custom models and probes, check out the **Custom Models Tutorial** notebook in the repository examples.

---

## Function Reference

Below is a listing of all public functions and classes in each module, along with their parameters and behavior.

### `pingkit.embedding` Module

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

### `pingkit.extraction` Module

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

### `pingkit.model` Module

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
    model: str = "mlp",
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
  * `model` (str or callable): Specifies the model to use. Built-in options include `"mlp"` and `"cnn"`. Alternatively, provide a callable to define custom models. If `"cnn"` or a custom CNN is used, `meta` must provided to reconstruct the input shape.
  * `meta`: Metadata dict (from `load_npz_features`) containing `"parts"`, `"layers"`, `"hidden_size"` (required if `model_type="cnn"`).
  * `num_classes`: Number of output classes.
  * `n_epochs`: Maximum number of epochs to train.
  * `learning_rate`: Optimizer learning rate.
  * `batch_size`: Mini‑batch size.
  * `device`: Compute device (`"cuda"` or `"cpu"`).
  * `contrastive_weight`: Weight λ for supervised contrastive loss (only used if `model_type="cnn"`).
  * `validation_data`: Tuple `(X_val, y_val)`. If provided, uses this as hold‑out validation set.
  * `val_split`: Fraction of data to set aside for validation (if `validation_data` is None).
  * `eval_metric`: Either a registered metric name ("roc_auc", "accuracy", "macro_f1", "loss") or a callable metric(y_true, y_prob). Default is `"roc_auc"`.
  * `class_weight` (str, sequence of floats, tensor, or None): Class weighting strategy ("balanced" or explicit class weights).
  * `loss_fn` (str or callable): Loss function to optimize, either `"ce"` (cross-entropy, default), `"focal"`, or a custom callable.
  * `early_stopping`: If True, enable early stopping on validation metric.
  * `patience`: Number of epochs with no improvement before stopping.
  * `random_state`: Seed for reproducibility (controls data shuffling).
* **Returns**: `(model, history)`:

  * `model`: Trained `nn.Module` (in eval mode).
  * `history`: List of dicts, each with keys `"epoch"`, `"train_loss"`, and `"val_metric"`.

> ⚠️ **Deprecation Notice:**

  * The parameter `model_type` is deprecated. Use `model` instead.
  * The parameter `metric` is deprecated. Use `eval_metric` instead.

#### `save_artifacts`

```python
from typing import Tuple

def save_artifacts(
    model: torch.nn.Module,
    *,
    path: str = "artifacts/pingkit",
    meta: dict | None = None,
) -> Tuple[str, str]:
```

* **Description**: Saves a trained `pingClassifier`, `pingContrastiveCNN` or custom model to disk: weights (`.pt`) and metadata (`.json`). The metadata includes model geometry (input\_dim, parts, layers, hidden\_size), hyperparameters (`n_examples`, `target_ratio`, `p_drop`, `width_cap`, and `proj_mult` for CNN), plus any additional user‑supplied `meta`.
* **Parameters**:

  * `model`: Trained PyTorch model instance (`pingClassifier` or `pingContrastiveCNN`).
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

#### `pingClassifier` Class

```python
class pingClassifier(nn.Module):
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

#### `pingContrastiveCNN` Class

```python
class pingContrastiveCNN(nn.Module):
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
  * `n_examples`, `target_ratio`, `width_cap`, `proj_mult`, `p_drop`: Same heuristics as `pingClassifier`, applied to CNN channel widths.
* **Attributes**:

  * `.encoder`: `pingCNNEncoder` instance.
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

Below is a complete example showing how to go from raw text prompts to embeddings, feature extraction, model training, evaluation, and plotting—using in‑memory DataFrames instead of reading directly from CSVs in each function call.

---

### 1. Prepare and format prompts

```python
import pandas as pd
from pingkit.embedding import embed_dataset

# Load raw prompts; must have columns ['id', 'prompt']
df = pd.read_csv('mmlu_prompts_ts.csv', index_col='id')

# Wrap each question in an instruction template
df['prompt'] = df['prompt'].apply(
    lambda x: (
        "<start_of_turn>user\n" + x + "<end_of_turn>\n"
        "<start_of_turn>model\nAnswer: "
    )
)
```

**What happens:**

* `df` is a `DataFrame` of shape `(n_samples, 1)`, indexed by `id`.
* Each `prompt` now looks like:

| id | prompt                                                                           |
| -- | -------------------------------------------------------------------------------- |
| q1 | `<start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\nAnswer: ` |
| q2 | `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\nAnswer: `          |

---

### 2. Extract token‑level embeddings

```python
embed_dataset(
    data=df,
    input_col='prompt',
    output_dir='mmlu_answer',
    model_name='google/gemma-2-9b-it',
    eos_token='<end_of_turn>',
    device='cuda:0',
    pooling='mean'
)
```

* Creates subdirectories under `mmlu_answer/`:

  ```
  mmlu_answer/rs/    # residual streams
  mmlu_answer/attn/  # attention outputs
  mmlu_answer/mlp/   # MLP outputs
  ```
* Within `rs/`, for example, each file `<id>_L<layer>.csv` is a column vector of shape `(hidden_size,)`.

---

### 3. Aggregate embeddings into a compressed NPZ

```python
from pingkit.extraction import extract_token_vectors
import os

layer = 35
npz_path = extract_token_vectors(
    embedding_dir='mmlu_answer',
    output_file=f'mmlu_answer/results/features_rs_L{layer}',
    layers=layer,
    parts='rs',
    n_jobs=os.cpu_count(),
)
print("✅   stacked features:", npz_path)
```

* **Output:** `mmlu_answer/results/features_rs_L35.npz`
* **Inside the NPZ:**

  * `data`: array of shape `(hidden_size, n_samples)`
  * `columns`: list of sample IDs (`n_samples` long)
  * Metadata: `parts=['rs']`, `layers=[35]`, `hidden_size` integer

---

### 4. Load features and raw labels

```python
from pingkit.model import load_npz_features
import pandas as pd

# Load the NPZ into a DataFrame
X_df, meta = load_npz_features(npz_path)
print(X_df.shape)    # e.g. (20000, 1024)
print(meta)          # e.g. {'parts': ['rs'], 'layers': [35], 'hidden_size': 1024}

# Load raw answers and map from letters to integers
y_raw = pd.read_csv('mmlu_g.csv', index_col='id')['answer']
mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
y_series = y_raw.map(mapping).fillna(0).astype(int)
```

* **Data shapes:**

  * `X_df`: `(n_samples, hidden_size)`
  * `y_series`: `(n_samples,)` with integer labels in `[0,3]`

---

### 5. Align and split into train/test

```python
from sklearn.model_selection import train_test_split

# Keep only samples present in both X and y
common = X_df.index.intersection(y_series.index)
X_df = X_df.loc[common]
y_series = y_series.loc[common]

# Stratified split: 5,000 examples for training
X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y_series,
    train_size=5000,
    stratify=y_series,
    shuffle=True,
    random_state=405,
)
print("Train:", X_train.shape, "Test:", X_test.shape)
```

* **Resulting shapes:**

  * Training: `(5000, hidden_size)`
  * Test: `(n_test, hidden_size)`

---

### 6. Train an MLP classifier

```python
model, history = fit(
    X_train,
    y_train.values,
    model_type='mlp',
    meta=meta,
    num_classes=4,
    metric='loss',
    batch_size=128,
    learning_rate=1e-2,
    contrastive_weight=0.4,
    n_epochs=100,
    val_split=0.2,
    early_stopping=True,
    random_state=405,
)
```

* **`history`:** List of dicts with keys:

  * `epoch`: epoch number
  * `train_loss`: training loss
  * `val_metric`: validation loss (since `metric='loss'`)

---

### 7. Save and reload model artifacts

```python
from pingkit.model import save_artifacts, load_artifacts

weights_path, meta_path = save_artifacts(
    model,
    path=f'artifacts/mmlu_rs_L{layer}',
    meta=meta
)
print("Saved:", weights_path, meta_path)

# Later…
model, meta = load_artifacts(f'artifacts/mmlu_rs_L{layer}', device='cuda')
```

* Saves `artifacts/mmlu_rs_L35.pt` and `.json` metadata

---

### 8. Evaluate on test set

```python
from pingkit.model import _evaluate
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import calibration_curve

# Prepare data for evaluation
device = next(model.parameters()).device
X_test_np = X_test.values.astype(np.float32)

probs, test_acc, _ = _evaluate(
    model,
    X_test_np,
    y_test.values,
    model_type='mlp',
    metric_fn=lambda y, p: accuracy_score(y, p.argmax(1)),
    device=device
)

auc = roc_auc_score(
    y_test.values,
    probs,
    multi_class='ovr',
    average='macro'
)
print(f"ACC {test_acc:.4f}   AUC {auc:.4f}")
```


---

## License

pingkit is released under the MIT License. See the [LICENSE](LICENSE) file for details.


## Contact

Jacob Berkowitz  ·  **[Jacob.Berkowitz2@cshs.org](mailto:Jacob.Berkowitz2@cshs.org)**
