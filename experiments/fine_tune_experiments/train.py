#!/usr/bin/env python3
import argparse
import re
import os
import json
from typing import List, Tuple
import torch
import pandas as pd

# Silence "forked after parallelism" warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
try:
    # AutoPeftModelForCausalLM auto-loads adapters if the folder contains PEFT weights
    from peft import AutoPeftModelForCausalLM
    HAS_AUTOPEFT = True
except Exception:
    HAS_AUTOPEFT = False

START_TURN = "<start_of_turn>"
END_TURN = "<end_of_turn>"

LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
ANSWER_KV_RE = re.compile(r"(?:^|\n)\s*(?:answer|final\s*answer)\s*[:\-\.]\s*([ABCD])\b", re.IGNORECASE)
REFUSAL_RE = re.compile(r"\b(sorry|cannot|can[’']?t)\b", re.IGNORECASE)


def extract_choice(text: str) -> str:
    """
    Prefer the first A/B/C/D that appears AFTER an 'Answer:' anchor.
    Then try 'Final Answer: X'.
    Finally, fall back to the first standalone letter (A-D).
    """
    s = "" if text is None else (text if isinstance(text, str) else str(text))

    # 1) After 'Answer:' anchor
    m_anchor = re.search(r"answer\s*:\s*", s, flags=re.IGNORECASE)
    if m_anchor:
        m = re.search(r"\b([ABCD])\b", s[m_anchor.end():], flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 2) 'Final Answer: X'
    m = ANSWER_KV_RE.search(s)
    if m:
        return m.group(1).upper()

    # 3) First standalone letter anywhere (last resort)
    m = LETTER_RE.search(s)
    if m:
        return m.group(1).upper()

    return ""


def normalize_gold(gold: str) -> str:
    """
    Normalize dataset 'answer' to A/B/C/D if possible; else return normalized text.
    """
    if gold is None:
        return ""
    gold = str(gold)
    m = ANSWER_KV_RE.search(gold) or LETTER_RE.search(gold)
    if m:
        return m.group(1).upper()
    return re.sub(r"\s+", " ", gold).strip().lower()


def load_model_and_tokenizer(
    model_path: str,
    base_model_name: str = "google/gemma-2-9b-it",
    load_in_4bit: bool = False,
    device: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Prefer tokenizer saved with your model dir; else fall back to base tokenizer
    tok_src = model_path if os.path.isdir(model_path) else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding is harmless here (we do one example at a time) but keeps behavior consistent
    tokenizer.padding_side = "right"

    quant_cfg = None
    device_map = None
    torch_dtype = torch.float16 if device == "cuda" else None

    if load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        device_map = {"": torch.cuda.current_device()} if device == "cuda" else None
    else:
        device_map = "auto" if device == "cuda" else None

    # Try AutoPeft first (adapter-only dirs). Fallback to base AutoModel if a full model was saved.
    if HAS_AUTOPEFT:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_cfg,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_cfg,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_cfg,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

    return model, tokenizer, device


def build_chat_prompt(tokenizer, user_text: str, answer_prefix: str = "Answer: ") -> str:
    """
    Construct the exact untokenized string fed to the model:
    - Use the tokenizer's chat template (Gemma IT), opening an assistant turn.
    - Append 'Answer: ' so decoding continues after it.
    DO NOT modify the user's prompt content itself.
    """
    messages = [{"role": "user", "content": user_text}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Only append if it's not already there (idempotent)
    if answer_prefix and not prompt_text.endswith(answer_prefix):
        prompt_text = prompt_text + answer_prefix
    return prompt_text


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    user_prompt: str,
    device: str,
    answer_prefix: str = "Answer: ",
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    debug_print: bool = False
) -> Tuple[str, List[int], List[int], str]:
    """
    Generate for a single user prompt and return:
      - decoded new tokens (string),
      - PADDED input_ids exactly fed to model,
      - PADDED attention_mask exactly fed to model,
      - the exact untokenized string used as input (chat template + prefix).
    """
    # Build templated input (no prompt edits) + seed prefix
    prompt_text = build_chat_prompt(tokenizer, user_prompt, answer_prefix)

    if debug_print:
        print("\n==== DEBUG: EXACT input to model (templated + prefix) ====")
        print(prompt_text)
        enc_dbg = tokenizer(prompt_text, return_tensors="pt")
        print(f"Tokens in prompt: {enc_dbg['input_ids'].shape[1]}")
        print("==== END DEBUG ====\n")

    enc = tokenizer(prompt_text, return_tensors="pt").to(device)

    # Prefer to stop at <end_of_turn>, fallback to eos
    stop_ids: List[int] = []
    eot_id = tokenizer.convert_tokens_to_ids(END_TURN)
    if isinstance(eot_id, int) and eot_id >= 0:
        stop_ids.append(eot_id)
    if isinstance(tokenizer.eos_token_id, int):
        stop_ids.append(tokenizer.eos_token_id)
    eos_setting = stop_ids if stop_ids else tokenizer.eos_token_id

    outputs = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=max(temperature, 1e-5),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_setting,
    )

    # Since we did ONE example, we can slice with the exact prompt length
    cut = enc["input_ids"].shape[1]
    gen = outputs[0, cut:]

    text = tokenizer.decode(gen, skip_special_tokens=True).strip()

    # If the model produced extra context before 'Answer:', trim to the first 'Answer:' anchor
    anchor = re.search(r"answer\s*:\s*", text, flags=re.IGNORECASE)
    if anchor:
        text = text[anchor.start():].lstrip()

    # If the model produced an explicit <end_of_turn>, strip it
    if text.endswith(END_TURN):
        text = text[: -len(END_TURN)].rstrip()

    # Copy inputs to CPU for saving
    input_ids_padded = enc["input_ids"].detach().cpu().tolist()[0]
    attn_mask_padded = enc["attention_mask"].detach().cpu().tolist()[0]

    return text, input_ids_padded, attn_mask_padded, prompt_text


def main():
    ap = argparse.ArgumentParser(description="Evaluate Gemma-2-IT LoRA model on medmcqa (single-example generation)")
    ap.add_argument("--model_path", type=str, required=True, help="Path to trained model dir (your --output_dir)")
    ap.add_argument("--data_csv", type=str, default="./medmcqa_train_sample_prompts.csv",
                    help="CSV with columns: id, subject, prompt, answer")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--save_predictions", type=str, default="./medmcqa_eval_predictions.csv")
    ap.add_argument("--report_json", type=str, default="./medmcqa_eval_report.json")
    ap.add_argument("--answer_prefix", type=str, default="Answer: ",
                    help="Prefix to seed at the start of the assistant's turn")
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only evaluate the first N rows (useful for quick smoke tests)")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.data_csv)
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)
    for col in ["id", "subject", "prompt", "answer"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.data_csv}")

    # Load model/tokenizer
    model, tokenizer, device = load_model_and_tokenizer(
        model_path=args.model_path,
        load_in_4bit=args.load_in_4bit,
    )
    model.eval()

    # Accumulators
    preds: List[str] = []
    choices: List[str] = []
    gold_norm: List[str] = []
    refusal_flags: List[int] = []
    inputs_used_text: List[str] = []
    inputs_ids_padded: List[List[int]] = []
    inputs_attn_mask: List[List[int]] = []
    inputs_lengths: List[int] = []

    # Iterate one example at a time
    for idx, row in df.iterrows():
        user_prompt = str(row["prompt"])
        gold = row["answer"]

        # Print debug only for the first example
        debug = (idx == 0)

        text, in_ids, in_mask, in_text = generate_one(
            model=model,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            device=device,
            answer_prefix=args.answer_prefix,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            debug_print=debug
        )

        preds.append(text if isinstance(text, str) else ("" if text is None else str(text)))
        inputs_used_text.append(in_text)
        inputs_ids_padded.append(in_ids)
        inputs_attn_mask.append(in_mask)
        inputs_lengths.append(int(sum(in_mask)))

        # Choice + gold normalization + refusal flag
        choices.append(extract_choice(preds[-1]))
        gold_norm.append(normalize_gold(gold))
        refusal_flags.append(1 if REFUSAL_RE.search(preds[-1] or "") else 0)

    # Compute correctness
    correct_flags = []
    for c, g, pred_text in zip(choices, gold_norm, preds):
        if g in {"A", "B", "C", "D"}:
            correct_flags.append(1 if c == g else 0)
        else:
            norm_pred = re.sub(r"\s+", " ", pred_text).strip().lower()
            correct_flags.append(1 if norm_pred == g else 0)

    df_out = pd.DataFrame({
        "id": df["id"],
        "subject": df["subject"],
        "prompt": df["prompt"],  # unchanged user prompt
        "gold_answer_raw": df["answer"],
        "gold_answer_norm": gold_norm,
        "model_text": preds,
        "model_choice": choices,
        "is_refusal": refusal_flags,
        "is_correct": correct_flags,
    })

    accuracy = sum(correct_flags) / max(1, len(correct_flags))
    by_subject = df_out.groupby("subject")["is_correct"].mean().reset_index().sort_values("is_correct", ascending=False)
    refusal_rate = sum(refusal_flags) / max(1, len(refusal_flags))

    # Save artifacts
    df_out.to_csv(args.save_predictions, index=False)
    report = {
        "num_examples": int(len(df_out)),
        "accuracy_overall": accuracy,
        "refusal_rate_overall": refusal_rate,
        "accuracy_by_subject": {str(r["subject"]): float(r["is_correct"]) for _, r in by_subject.iterrows()},
        "model_path": args.model_path,
        "data_csv": args.data_csv,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "answer_prefix": args.answer_prefix,
        "tokenizer_chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
        # exact inputs used (templated text + prefix), plus tensors as JSON
        "model_input_text": inputs_used_text,
        "model_input_ids_padded": [json.dumps(x) for x in inputs_ids_padded],
        "model_attention_mask": [json.dumps(x) for x in inputs_attn_mask],
        "model_input_length": inputs_lengths,
    }
    with open(args.report_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Examples: {len(df_out)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Refusal rate: {refusal_rate:.4f}")
    print("Top subjects:")
    for _, r in by_subject.head(10).iterrows():
        print(f"  {r['subject']}: {r['is_correct']:.4f}")
    print(f"Predictions saved to: {args.save_predictions}")
    print(f"Report saved to: {args.report_json}")


if __name__ == "__main__":
    main()
