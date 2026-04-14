# generate.py

from __future__ import annotations
import logging, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from .embedding import (
    load_model_and_tokenizer,
    _register_block_hooks,
    _primary_device,
    _pooled,
    _find_target_token,
    _SKIP_PUNCT_RE,
    _SKIP_DUP_RE,
)
from .model import load_artifacts

LOGGER = logging.getLogger(__name__)
__all__ = ["ProbeSpec", "generate_with_probes"]


@dataclass
class ProbeSpec:
    name: str
    model_path: Optional[str] = None
    model: Optional[torch.nn.Module] = None
    meta: Optional[dict] = None
    parts: Optional[Sequence[str]] = None
    layers: Optional[Sequence[int]] = None
    pooling: str = "last"
    class_index: int = 1
    device: Union[str, torch.device] = "auto"


def _valid_token_indices(tokens: List[str], *, eos_token: Optional[str]) -> List[int]:
    idxs: List[int] = []
    for i, tok in enumerate(tokens):
        if _SKIP_PUNCT_RE.match(tok) or _SKIP_DUP_RE.match(tok):
            continue
        if eos_token is not None and eos_token in tok:
            continue
        idxs.append(i)
    if not idxs and tokens:
        return list(range(len(tokens)))
    return idxs


def _safe_valid_idxs(idxs: List[int], length: int) -> List[int]:
    if length <= 0:
        return []
    clamped = [i for i in idxs if 0 <= i < length]
    if not clamped:
        clamped = [length - 1]
    return clamped


def _build_feature_for_probe(
    *,
    probe_parts: Sequence[str],
    probe_layers: Sequence[int],
    pooling: str,
    tokens: List[str],
    valid_idxs: List[int],
    hidden_states: Tuple[torch.Tensor, ...],
    attn_cache: Dict[int, torch.Tensor],
    mlp_cache: Dict[int, torch.Tensor],
) -> np.ndarray:
    vecs: List[np.ndarray] = []
    for layer in probe_layers:
        seq_rs = hidden_states[layer + 1][0].cpu()
        seq_attn = attn_cache[layer].cpu()
        seq_mlp = mlp_cache[layer].cpu()

        local_idxs_rs = _safe_valid_idxs(valid_idxs, seq_rs.size(0))
        local_idxs_attn = _safe_valid_idxs(valid_idxs, seq_attn.size(0))
        local_idxs_mlp = _safe_valid_idxs(valid_idxs, seq_mlp.size(0))

        pool_rs = _pooled(seq_rs, local_idxs_rs, pooling).numpy()
        pool_attn = _pooled(seq_attn, local_idxs_attn, pooling).numpy()
        pool_mlp = _pooled(seq_mlp, local_idxs_mlp, pooling).numpy()

        for part in probe_parts:
            if part == "rs":
                vecs.append(pool_rs)
            elif part == "attn":
                vecs.append(pool_attn)
            elif part == "mlp":
                vecs.append(pool_mlp)
            else:
                raise ValueError(f"Unknown part {part!r}")
    return np.concatenate(vecs, axis=0).astype(np.float32)


def _forward_logits(model: torch.nn.Module, xb: torch.Tensor) -> torch.Tensor:
    out = model(xb)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    return logits


def _ensure_probe_loaded(probe: ProbeSpec) -> Tuple[torch.nn.Module, dict]:
    if probe.model is not None and probe.meta is not None:
        return probe.model, probe.meta
    if probe.model_path is None:
        raise ValueError(f"Probe {probe.name!r} needs either model+meta or model_path.")
    m, meta = load_artifacts(probe.model_path, device="cpu")
    return m, meta


def _probs_for_feature(
    probe_model: torch.nn.Module,
    feat: np.ndarray,
    class_index: int,
    device: torch.device,
) -> float:
    with torch.no_grad():
        xb = torch.tensor(feat[None, :], device=device)
        logits = _forward_logits(probe_model, xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    class_index = int(class_index)
    if class_index < 0 or class_index >= probs.size:
        raise ValueError(f"class_index {class_index} out of range for {probs.size} classes")
    return float(probs[class_index])


def _encode_prompt(tokenizer: PreTrainedTokenizer, text: str, device: torch.device):
    enc = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def _sample_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> int:
    if not do_sample:
        return int(torch.argmax(logits, dim=-1))
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum <= top_p
        mask[..., 0] = True
        capped_probs = sorted_probs * mask
        capped_probs = capped_probs / capped_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(capped_probs, num_samples=1)
        token_id = sorted_idx.gather(-1, choice).item()
        return int(token_id)
    token_id = torch.multinomial(probs, num_samples=1).item()
    return int(token_id)


def generate_with_probes(
    prompt: str,
    *,
    probes: Sequence[ProbeSpec],
    evaluate_on: str = "first",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    model_name: Optional[str] = None,
    device: Union[str, torch.device] = "auto",
    quantization: Optional[str] = None,
    lora_adapter: Optional[str] = None,
    merge_lora: bool = False,
    attn_implementation: Optional[str] = None,
    generation_kwargs: Optional[dict] = None,
    eos_token: Optional[str] = None,
    filter_non_text: bool = True,
    return_details: bool = True,
) -> dict:
    if model is None or tokenizer is None:
        if model_name is None:
            model_name = "Qwen/Qwen3-0.6B"
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            quantization=quantization,
            device_map="auto" if str(device) == "auto" else None,
            lora_adapter=lora_adapter,
            merge_lora=merge_lora,
            attn_implementation=attn_implementation,
        )
    model.eval()

    handles, attn_cache, mlp_cache = _register_block_hooks(model)
    dev = _primary_device(model)

    results: Dict[str, dict] = {p.name: {} for p in probes}
    gen_conf = dict(max_new_tokens=128, do_sample=False, temperature=1.0, top_p=1.0)
    if generation_kwargs:
        gen_conf.update(generation_kwargs)

    try:
        if evaluate_on not in {"first", "stream", "last"}:
            raise ValueError("evaluate_on must be 'first', 'stream', or 'last'.")

        enc = _encode_prompt(tokenizer, prompt, dev)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        tokens_in = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())

        if filter_non_text:
            valid_idxs_in = _valid_token_indices(tokens_in, eos_token=eos_token)
        else:
            valid_idxs_in = list(range(len(tokens_in)))

        loaded: Dict[str, Tuple[torch.nn.Module, dict]] = {}
        for p in probes:
            pm, meta = _ensure_probe_loaded(p)
            pm = pm.to(dev).eval()
            loaded[p.name] = (pm, meta)

        if evaluate_on in {"first", "stream"}:
            for p in probes:
                pm, meta = loaded[p.name]
                parts = tuple((p.parts or meta.get("parts", ("rs", "attn", "mlp"))))
                layers = tuple((p.layers or meta.get("layers", ())))
                if not layers:
                    layers = tuple(sorted(attn_cache.keys()))
                feat = _build_feature_for_probe(
                    probe_parts=parts,
                    probe_layers=layers,
                    pooling=p.pooling,
                    tokens=tokens_in,
                    valid_idxs=valid_idxs_in,
                    hidden_states=out.hidden_states,
                    attn_cache=attn_cache,
                    mlp_cache=mlp_cache,
                )
                prob = _probs_for_feature(pm, feat, p.class_index, dev)
                results[p.name]["first"] = prob

        generated_ids: List[int] = []
        stream_probs: Dict[str, List[float]] = {p.name: [] for p in probes}

        if evaluate_on in {"stream", "last"}:
            use_cache = True
            past_key_values = None
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", None)

            max_new = int(gen_conf.get("max_new_tokens", 128))
            do_sample = bool(gen_conf.get("do_sample", False))
            temperature = float(gen_conf.get("temperature", 1.0))
            top_p = float(gen_conf.get("top_p", 1.0))
            eos_id = gen_conf.get("eos_token_id", getattr(tokenizer, "eos_token_id", None))

            for t in range(max_new):
                with torch.no_grad():
                    out_step = model(
                        input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        output_hidden_states=True,
                    )
                logits = out_step.logits[:, -1, :]
                past_key_values = getattr(out_step, "past_key_values", None)

                next_id = _sample_next_token(
                    logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
                if past_key_values is None:
                    input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=dev)], dim=1)
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
                        )
                else:
                    input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=dev)], dim=1)
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
                        )
                generated_ids.append(next_id)

                tokens_all = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
                if filter_non_text:
                    valid_idxs = _valid_token_indices(tokens_all, eos_token=eos_token)
                else:
                    valid_idxs = list(range(len(tokens_all)))

                if evaluate_on == "stream":
                    for p in probes:
                        pm, meta = loaded[p.name]
                        parts = tuple((p.parts or meta.get("parts", ("rs", "attn", "mlp"))))
                        layers = tuple((p.layers or meta.get("layers", ())))
                        if not layers:
                            layers = tuple(sorted(attn_cache.keys()))
                        feat = _build_feature_for_probe(
                            probe_parts=parts,
                            probe_layers=layers,
                            pooling=p.pooling,
                            tokens=tokens_all,
                            valid_idxs=valid_idxs,
                            hidden_states=out_step.hidden_states,
                            attn_cache=attn_cache,
                            mlp_cache=mlp_cache,
                        )
                        prob = _probs_for_feature(pm, feat, p.class_index, dev)
                        stream_probs[p.name].append(prob)

                if eos_id is not None and next_id == int(eos_id):
                    break

            text_out = tokenizer.decode(generated_ids, skip_special_tokens=True)

            if evaluate_on == "stream":
                for p in probes:
                    results[p.name]["stream"] = stream_probs[p.name]

            if evaluate_on in {"last"}:
                enc_full = {"input_ids": input_ids, "attention_mask": attention_mask} if attention_mask is not None else {"input_ids": input_ids}
                with torch.no_grad():
                    out_full = model(**enc_full, output_hidden_states=True)
                tokens_all = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
                if filter_non_text:
                    valid_idxs = _valid_token_indices(tokens_all, eos_token=eos_token)
                else:
                    valid_idxs = list(range(len(tokens_all)))
                for p in probes:
                    pm, meta = loaded[p.name]
                    parts = tuple((p.parts or meta.get("parts", ("rs", "attn", "mlp"))))
                    layers = tuple((p.layers or meta.get("layers", ())))
                    if not layers:
                        layers = tuple(sorted(attn_cache.keys()))
                    feat = _build_feature_for_probe(
                        probe_parts=parts,
                        probe_layers=layers,
                        pooling=p.pooling,
                        tokens=tokens_all,
                        valid_idxs=valid_idxs,
                        hidden_states=out_full.hidden_states,
                        attn_cache=attn_cache,
                        mlp_cache=mlp_cache,
                    )
                    prob = _probs_for_feature(pm, feat, p.class_index, dev)
                    results[p.name]["last"] = prob

            return {
                "text": text_out,
                "probes": results if return_details else {k: {m: v[m] for m in results[k] if m in {"first", "last"}} for k in results},
            }

        return {
            "text": "",
            "probes": results if return_details else {k: {"first": results[k].get("first", math.nan)} for k in results},
        }

    finally:
        for h in handles:
            h.remove()
