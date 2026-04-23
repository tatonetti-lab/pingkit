import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── settings ──────────────────────────────────────────────────────────────────
MODEL_NAME      = "google/gemma-2-9b-it"
INPUT           = "mmlu_prompts_gemma_train.csv"
OUTPUT          = "mmlu_gemma_9b_train_CoT_prompts.csv"
COT_MAX_TOKENS  = 1024          # max tokens for the reasoning trace
COT_SUFFIX      = "\nLet's think through this step by step:\n"
ANSWER_SUFFIX   = "\nTherefore, the letter of the correct answer is **("

# ── load model ────────────────────────────────────────────────────────────────
print("Loading model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",          # shard layers across all available GPUs
    torch_dtype=torch.float16,  # half-precision to save VRAM
)
model.eval()

# ── helper ────────────────────────────────────────────────────────────────────
def generate_cot(prompt: str) -> str:
    """
    Generate a chain-of-thought reasoning trace (greedy / deterministic).
    """
    cot_prompt = prompt + COT_SUFFIX
    inputs = tokenizer(cot_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=COT_MAX_TOKENS,
            do_sample=False,           # greedy for reproducibility
        )
    # decode only the newly generated tokens (strip the input prompt tokens)
    generated_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

# ── main loop ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT)
# Strip any existing trailing "Answer:" so we control the format
df["prompt"] = df["prompt"].str.replace(r"\s*Answer:\s*$", "", regex=True)

records = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="id"):
    base_prompt = row["prompt"]

    # Stage 1: generate chain-of-thought reasoning
    cot_text = generate_cot(base_prompt)

    # Stage 2: assemble full prompt ending at "**" (no answer token)
    full_prompt = base_prompt + COT_SUFFIX + cot_text + ANSWER_SUFFIX

    tqdm.write(f"{row['id']}: prompt ready ({len(full_prompt)} chars)")

    records.append({
        "id"      : row["id"],
        "subject" : row["subject"],
        "correct" : row["answer"].strip(),
        "prompt"  : full_prompt,        # ends with "**", ready for answer prediction
    })

pd.DataFrame(records).to_csv(OUTPUT, index=False)
print(f"Saved results ➜  {OUTPUT}")