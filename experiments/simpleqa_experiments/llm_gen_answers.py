import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── settings ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-9b-it"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
INPUT      = "simpleqa_prompts.csv"
OUTPUT     = "simpleqa_results.csv"
MAX_NEW_TOKENS = 64  # adjust as needed for answer length

# ── load model ────────────────────────────────────────────────────────────────
print(f"Loading model on {DEVICE}…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ── helper ────────────────────────────────────────────────────────────────────
def generate_answer(prompt: str) -> tuple[str, str]:
    """
    Generate a response to the prompt.
    Returns (answer, full_text) where:
      - answer: just the newly generated text
      - full_text: the complete prompt + generated response
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
    
    # Decode full output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Decode just the new tokens (the answer)
    answer_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    return answer, full_text

# ── main loop ─────────────────────────────────────────────────────────────────
print(f"Loading data from {INPUT}…")
df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} rows")

records = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
    answer, full_text = generate_answer(row["prompt"])
    
    tqdm.write(f"{row['id']}: {answer[:50]}...")
    
    records.append({
        "id":         row["id"],
        "metadata":   row["metadata"],
        "problem":    row["problem"],
        "true_answer": row["true_answer"],
        "prompt":     row["prompt"],
        "answer":     answer,
        "full_text":  full_text,
    })

result_df = pd.DataFrame(records)
result_df.to_csv(OUTPUT, index=False)
print(f"\nSaved {len(result_df)} results → {OUTPUT}")

# ── preview ───────────────────────────────────────────────────────────────────
print("\n--- Sample result ---")
sample = result_df.iloc[0]
print(f"ID: {sample['id']}")
print(f"Problem: {sample['problem']}")
print(f"True Answer: {sample['true_answer']}")
print(f"Model Answer: {sample['answer']}")
print(f"Full Text:\n{sample['full_text']}")