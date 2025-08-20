from __future__ import annotations
import os, re, glob, logging, typing
from typing import Sequence, Tuple, Dict, Optional, Union, List
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
LOGGER = logging.getLogger(__name__)

__all__ = ["extract_token_vectors"]

_FILE_RE = re.compile(r"(?P<qid>.+?)_L(?P<layer>\d+)\.csv$")


# helpers
def _discover_basic_dims(embed_dir: str, part: str) -> Tuple[int, int, List[str]]:
    files = glob.glob(os.path.join(embed_dir, part, "*_L*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {os.path.join(embed_dir, part)}")
    qids = sorted({_FILE_RE.match(os.path.basename(f)).group("qid") for f in files})
    first_qid_files = [f for f in files if _FILE_RE.match(os.path.basename(f)).group("qid") == qids[0]]
    n_layers = len(first_qid_files)
    hidden_size = pd.read_csv(first_qid_files[0]).shape[0]
    return n_layers, hidden_size, qids


def _read_vector(path: str, expected_len: int) -> np.ndarray:
    """
    Read a single embedding vector from CSV and return a 1-D float32 array
    of length == expected_len. If the file is empty or malformed, return a
    NaN-filled array of that length.
    """
    try:
        df = pd.read_csv(path, engine="python")
    except Exception as e:
        LOGGER.warning("Failed to read %s (%s). Filling with NaNs.", path, e)
        return np.full(expected_len, np.nan, dtype=np.float32)

    if df.empty:
        LOGGER.warning("Empty CSV: %s. Filling with NaNs.", path)
        return np.full(expected_len, np.nan, dtype=np.float32)

    # Prefer already-numeric columns; otherwise coerce all to numeric
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        num = df.apply(pd.to_numeric, errors="coerce")

    # Heuristic: take the last numeric column as the embedding values
    # (adjust if you know the exact column name/index)
    try:
        s = num.iloc[:, -1]
    except Exception:
        LOGGER.warning("No numeric columns found in %s. Filling with NaNs.", path)
        return np.full(expected_len, np.nan, dtype=np.float32)

    vec = s.to_numpy(dtype=np.float32, copy=False)

    # Enforce expected length: pad/truncate with NaNs to avoid downstream crashes
    if vec.size < expected_len:
        vec = np.pad(vec, (0, expected_len - vec.size), constant_values=np.nan)
        LOGGER.warning("Padded %s from %d to %d with NaNs.", path, vec.size, expected_len)
    elif vec.size > expected_len:
        LOGGER.warning("Truncated %s from %d to %d.", path, vec.size, expected_len)
        vec = vec[:expected_len]

    return vec




def _process_qid(qid: str, embed_dir: str, parts: Sequence[str],
                 layers: Sequence[int], hidden_size: int) -> Tuple[str, np.ndarray]:
    vecs: List[np.ndarray] = []
    for part in parts:
        for layer in layers:
            fname = f"{qid}_L{layer:02d}.csv"
            path = os.path.join(embed_dir, part, fname)
            if os.path.isfile(path):
                v = _read_vector(path, expected_len=hidden_size)
                vecs.append(v)
            else:
                LOGGER.warning("Missing embedding file: %s (filled with NaNs)", path)
                vecs.append(np.full(hidden_size, np.nan, dtype=np.float32))
    return qid, np.concatenate(vecs, axis=0)



# public
def extract_token_vectors(
    embedding_dir: str,
    *,
    parts: Union[str, Sequence[str]] = ("rs", "attn", "mlp"),
    layers: Union[int, Sequence[int], None] = None,
    output_file: Optional[str] = None,
    save_csv: bool = False,
    n_jobs: int = 8,
) -> str:
    if isinstance(parts, str):
        parts = (parts,)
    parts = tuple(parts)

    LOGGER.info("Discovering embedding dimensions …")
    n_layers_total, hidden_size, qids = _discover_basic_dims(embedding_dir, parts[0])

    if layers is None:
        layers = list(range(n_layers_total))
    elif isinstance(layers, int):
        layers = [layers]
    layers = list(layers)

    LOGGER.info(
        "Collapse → parts=%s  layers=%s  hidden=%d  samples=%d",
        parts, layers, hidden_size, len(qids)
    )

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_qid)(qid, embedding_dir, parts, layers, hidden_size)
        for qid in tqdm(qids, desc="Stacking embeddings", unit="qid")
    )
    data: Dict[str, np.ndarray] = dict(results)

    for qid, arr in data.items():
        if arr.size != hidden_size * len(parts) * len(layers):
            LOGGER.warning(
                "Vector size mismatch for %s: got %d, expected %d",
                qid, arr.size, hidden_size * len(parts) * len(layers)
            )

    
    # each col is a sample id, row-index is feature
    feature_count = hidden_size * len(parts) * len(layers)
    df = pd.DataFrame(data, index=range(feature_count), dtype=np.float32)

    # naming
    if output_file is None:
        parts_str = "_".join(parts)
        layers_str = "-".join(f"{l:02d}" for l in layers)
        output_file = os.path.join(embedding_dir, "results",
                                   f"{parts_str}_L{layers_str}_stacked.npz")
    elif not output_file.endswith(".npz"):
        output_file += ".npz"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # save
    np.savez_compressed(
        output_file,
        data=df.values.astype(np.float32),
        columns=df.columns.values,
        parts=np.asarray(parts),
        layers=np.asarray(layers, dtype=np.int16),
        hidden_size=np.asarray([hidden_size], dtype=np.int16),
    )
    LOGGER.info("Saved compressed NPZ → %s", output_file)

    if save_csv:
        csv_path = output_file.rsplit(".", 1)[0] + ".csv"
        df.T.to_csv(csv_path, index=True, float_format="%.6g")  # rows=id, cols=feat
        LOGGER.info("Also saved CSV (transposed) → %s", csv_path)

    return output_file


if __name__ == "__main__":
    extract_token_vectors(
        "embeddings",
        parts=("rs", "mlp", "attn"),
        layers=[20],
        output_file="embeddings/features_hidden_mlp_L00-01-07.npz",
        save_csv=True,
    )
