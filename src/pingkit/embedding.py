from __future__ import annotations
import os, re, logging
from typing import Iterable, Union, Dict, List
import numpy as np
import torch, pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger(__name__)
__all__ = ["load_model_and_tokenizer", "embed_dataset", "embed"]

# skip pure punctuation/symbol tokens
# this is for the pooling function in case user wants to drop potentially irrelevant tokens
_SKIP_PUNCT_RE = re.compile(r"^[^A-Za-z0-9]+$")
# skip pandas duplicate columns: optional non-alnum → dot → digits
_SKIP_DUP_RE   = re.compile(r"^[^A-Za-z0-9]*\.[0-9]+$")


# helpers

def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return underlying model if wrapped in DataParallel/DDP."""
    if isinstance(model, (torch.nn.DataParallel,
                          torch.nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def _primary_device(model: torch.nn.Module) -> torch.device:
    """Device where inputs should live (first param/buffer → safe)."""
    m = _unwrap(model)
    for t in m.parameters(recurse=True):
        return t.device
    for t in m.buffers(recurse=True):
        return t.device
    return torch.device("cpu")


def _get_blocks(model: torch.nn.Module):
    """Locate transformer blocks in many HF architectures."""
    m = _unwrap(model)
    if hasattr(m, "layers"):
        return m.layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "encoder") and hasattr(m.encoder, "layer"):
        return m.encoder.layer
    if hasattr(m, "decoder") and hasattr(m.decoder, "layers"):
        return m.decoder.layers
    raise AttributeError("Cannot locate transformer layers.")


def _get_attn_and_mlp(block: torch.nn.Module):
    """Return (attention, mlp) sub-modules regardless of naming."""
    attn_names = ("self_attn", "attn", "attention", "self_attention")
    mlp_names  = ("mlp", "ffn", "feed_forward", "feedforward")
    attn = next((getattr(block, n) for n in attn_names if hasattr(block, n)), None)
    mlp  = next((getattr(block, n) for n in mlp_names  if hasattr(block, n)), None)
    if attn is None or mlp is None:
        raise AttributeError(f"Cannot find attn/mlp in {type(block).__name__}")
    return attn, mlp


# load

def load_model_and_tokenizer(
    model_name: str = "google/gemma-2b-it",
    *,
    quantization: str | None = None,
    device_map: str | None = "auto",   # keep "auto" so big models shard
):
    """Load HF model + tokenizer without DataParallel wrapping."""
    LOGGER.info("Loading model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quantization == "4bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, load_in_4bit=True, device_map=device_map
        )
    elif quantization == "8bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, load_in_8bit=True, device_map=device_map
        )
    else:
        model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, device_map=device_map
        ).eval()

    return model, tokenizer


# hooks

def _register_block_hooks(model) -> tuple[
    List[torch.utils.hooks.RemovableHandle],
    Dict[int, torch.Tensor],
    Dict[int, torch.Tensor],
]:
    """Attach hooks capturing attention & MLP outputs."""
    attn_cache, mlp_cache, handles = {}, {}, []

    for idx, block in enumerate(_get_blocks(model)):
        attn_sub, mlp_sub = _get_attn_and_mlp(block)

        def make_hook(cache, i):
            def hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                cache[i] = t.detach().cpu().squeeze(0)
            return hook

        handles.append(attn_sub.register_forward_hook(make_hook(attn_cache, idx)))
        handles.append(mlp_sub .register_forward_hook(make_hook(mlp_cache , idx)))

    return handles, attn_cache, mlp_cache


# token helpers
# not used for now but maybe helpful later. I just added the logic into the actual functions
def _find_target_token(tokens: List[str]) -> tuple[int, str]:
    """Last meaningful token (skip eos, punct, pandas dup suffix)."""
    for i in range(len(tokens) - 1, -1, -1):
        tok = tokens[i]
        if "end_of_turn" in tok or _SKIP_PUNCT_RE.match(tok) or _SKIP_DUP_RE.match(tok):
            continue
        return i, tok
    return len(tokens) - 1, tokens[-1]


def _pooled(t: torch.Tensor, idxs: List[int], method: str) -> torch.Tensor:
    """
    Return a 1-D pooled vector from a 2-D [seq_len, hidden] tensor.
    `idxs` are indices of valid tokens (already filtered).
    """
    if not idxs:                         # degenerate: fall back to all
        idxs = list(range(t.size(0)))

    if method == "last":
        return t[idxs[-1]]
    elif method == "first":
        return t[idxs[0]]
    elif method == "mean":
        return t[idxs].mean(dim=0)
    elif method == "max":
        return t[idxs].max(dim=0).values
    else:
        raise ValueError(f"Unknown pooling method '{method}'")


def _pool_key(method: str, tokens: List[str], valid_idxs: List[int]) -> str:
    """
    Column / dict key to use for a given pooling method.
    • "last"  → last valid *token string*
    • "first" → first valid *token string*
    • others  → method name itself
    """
    if method == "last":
        return tokens[valid_idxs[-1]]
    if method == "first":
        return tokens[valid_idxs[0]]
    return method


# main functions

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
):
    """
    Extract embeddings with multiple pooling strategies and save CSVs.
    If filter_non_text is True, skip pure-punct/symbol tokens, pandas-duplicate suffixes,
    and any token containing "end_of_turn". Otherwise include all tokens.
    """
    # ------------- load / setup -------------
    if isinstance(data, (str, os.PathLike)):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame({"__input__": list(data)})
        input_col = "__input__"
    if input_col is None:
        raise ValueError("input_col required when data is DataFrame")

    if isinstance(pooling, str):
        pooling = [pooling]

    model, tokenizer = load_model_and_tokenizer(model_name, device_map=device)
    n_layers = len(_get_blocks(model))
    layers   = layers or list(range(n_layers))
    parts    = parts  or ["rs", "attn", "mlp"]

    handles, attn_cache, mlp_cache = _register_block_hooks(model)

    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
            prompt = row[input_col] #+ (eos_token or "")
            enc    = tokenizer(prompt, return_tensors="pt")
            dev    = _primary_device(model)
            enc    = {k: v.to(dev) for k, v in enc.items()}

            attn_cache.clear(); mlp_cache.clear()
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)

            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())

            if filter_non_text:
                # filtering behavior
                valid_idxs = [
                    i for i, tok in enumerate(tokens)
                    if not _SKIP_PUNCT_RE.match(tok)
                       and not _SKIP_DUP_RE.match(tok)
                       and eos_token not in tok
                ]
            else:
                # include all tokens
                valid_idxs = list(range(len(tokens)))

            row_id = row.get("id", f"row_{idx}")

            for layer in layers:
                if not (0 <= layer < n_layers):
                    raise ValueError(f"Layer {layer} out of 0-{n_layers-1}")

                seq_rs   = out.hidden_states[layer + 1][0].cpu()
                seq_attn = attn_cache[layer].cpu()
                seq_mlp  = mlp_cache[layer].cpu()

                for method in pooling:
                    key_name = _pool_key(method, tokens, valid_idxs)
                    vecs = {
                        "rs":   _pooled(seq_rs,   valid_idxs, method),
                        "attn": _pooled(seq_attn, valid_idxs, method),
                        "mlp":  _pooled(seq_mlp,  valid_idxs, method),
                    }
                    for part in parts:
                        path_dir = os.path.join(output_dir, part)
                        os.makedirs(path_dir, exist_ok=True)
                        path = os.path.join(path_dir, f"{row_id}_L{layer:02d}.csv")
                        pd.DataFrame(
                            vecs[part].numpy().reshape(-1, 1),
                            columns=[key_name]
                        ).to_csv(path, index=False)
    finally:
        for h in handles: h.remove()


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
    """
    Return embeddings with multiple pooling strategies.

    If filter_non_text is True, skip punct/symbol tokens, pandas-duplicate suffixes,
    and "end_of_turn"; otherwise include all tokens so first/last map to true ends.
    """
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(pooling, str):
        pooling = [pooling]

    model, tokenizer = load_model_and_tokenizer(model_name, device_map=device)
    n_layers = len(_get_blocks(model))
    layers   = layers or list(range(n_layers))
    parts    = parts  or ["rs", "attn", "mlp"]

    handles, attn_cache, mlp_cache = _register_block_hooks(model)
    embeddings: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]] = {}

    try:
        for s in tqdm(inputs, desc="Embedding"):
            prompt = s #+ (eos_token or "")
            enc    = tokenizer(prompt, return_tensors="pt")
            dev    = _primary_device(model)
            enc    = {k: v.to(dev) for k, v in enc.items()}

            attn_cache.clear(); mlp_cache.clear()
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)

            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze())

            if filter_non_text:
                valid_idxs = [
                    i for i, tok in enumerate(tokens)
                    if not _SKIP_PUNCT_RE.match(tok)
                       and not _SKIP_DUP_RE.match(tok)
                       and eos_token not in tok
                ]
            else:
                valid_idxs = list(range(len(tokens)))

            embeddings[s] = {}
            for layer in layers:
                seq_rs   = out.hidden_states[layer + 1][0].cpu()
                seq_attn = attn_cache[layer].cpu()
                seq_mlp  = mlp_cache[layer].cpu()

                part_pool: Dict[str, Dict[str, np.ndarray]] = {p: {} for p in parts}
                for method in pooling:
                    key_name = _pool_key(method, tokens, valid_idxs)
                    part_pool["rs"][key_name]   = _pooled(seq_rs,   valid_idxs, method).numpy()
                    part_pool["attn"][key_name] = _pooled(seq_attn, valid_idxs, method).numpy()
                    part_pool["mlp"][key_name]  = _pooled(seq_mlp,  valid_idxs, method).numpy()
                embeddings[s][layer] = part_pool
    finally:
        for h in handles: h.remove()

    return embeddings
