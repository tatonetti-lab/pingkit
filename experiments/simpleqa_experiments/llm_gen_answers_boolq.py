import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── settings ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-9b-it"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
INPUT      = "boolq_prompts.csv"
OUTPUT     = "boolq_results.csv"
MAX_NEW_TOKENS = 64  # adjust as needed for answer length

# ── load model ────────────────────────────────────────────────────────────────
print(f"Loading model on {DEVICE}…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ── helper ────────────────────────────────────────────────────────────────────
def generate_answer(prompt: str) -> tuple[str, str, float]:
    """
    Generate a response to the prompt.
    Returns (answer, full_text, perplexity) where:
      - answer: just the newly generated text
      - full_text: the complete prompt + generated response
      - perplexity: per-token perplexity of the generated response tokens (conditioned on the prompt)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy decoding for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )

    # Compute perplexity over generated tokens (conditioned on the prompt)
    with torch.no_grad():
        full_ids = outputs[0].unsqueeze(0)  # [1, seq_len]
        attention_mask = torch.ones_like(full_ids, device=DEVICE)

        logits = model(input_ids=full_ids, attention_mask=attention_mask).logits  # [1, seq_len, vocab]
        shift_logits = logits[:, :-1, :]     # predicts tokens 1..end
        shift_labels = full_ids[:, 1:]       # actual tokens 1..end

        # Generated tokens are at positions [input_length .. seq_len-1] in full_ids
        # In shift_labels space, those correspond to indices [input_length-1 .. seq_len-2]
        start = input_length - 1
        answer_logits = shift_logits[:, start:, :]
        answer_labels = shift_labels[:, start:]

        # Optionally drop a trailing EOS from perplexity calculation
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and answer_labels.numel() > 0 and answer_labels[0, -1].item() == eos_id:
            answer_logits = answer_logits[:, :-1, :]
            answer_labels = answer_labels[:, :-1]

        if answer_labels.numel() == 0:
            perplexity = float("nan")
        else:
            log_probs = torch.log_softmax(answer_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, answer_labels.unsqueeze(-1)).squeeze(-1)
            nll = -token_log_probs.mean()
            perplexity = torch.exp(nll).item()
    
    # Decode full output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Decode just the new tokens (the answer)
    answer_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    return answer, full_text, perplexity

# ── main loop ─────────────────────────────────────────────────────────────────
print(f"Loading data from {INPUT}…")
df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} rows")

records = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
    answer, full_text, perplexity = generate_answer(row["prompt"])
    
    tqdm.write(f"{row['id']}: {answer[:50]}... (ppl={perplexity:.3f})")
    
    records.append({
        "id":         row["id"],
        "question":    row["question"],
        "true_answer": row["true_answer"],
        "prompt":     row["prompt"],
        "answer":     answer,
        "full_text":  full_text,
        "perplexity": perplexity,
    })

result_df = pd.DataFrame(records)
result_df.to_csv(OUTPUT, index=False)
print(f"\nSaved {len(result_df)} results → {OUTPUT}")

# ── preview ───────────────────────────────────────────────────────────────────
print("\n--- Sample result ---")
sample = result_df.iloc[0]
print(f"ID: {sample['id']}")
print(f"question: {sample['question']}")
print(f"True Answer: {sample['true_answer']}")
print(f"Model Answer: {sample['answer']}")
print(f"Perplexity: {sample['perplexity']}")
print(f"Full Text:\n{sample['full_text']}")
