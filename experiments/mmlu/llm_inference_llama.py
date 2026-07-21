"""
Evaluate the log-probability of the single most-likely next token for each
prompt in `INPUT` using Meta-Llama-3-70B-Instruct, writing the result
out to `OUTPUT`.

Works on multiple GPUs without running out of host RAM.
"""

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── settings ────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
INPUT      = "mmlu_prompts_llama_test.csv"
OUTPUT     = "mmlu_llama70_test_top_token.csv"

# ── load model & tokenizer ──────────────────────────────────────────────────
print("⏳  Loading model…  This can take a minute on first run.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",        # automatically shard across all visible GPUs
    torch_dtype=torch.float16 # halves memory footprint vs fp32
    # attn_implementation="flash_attention_2",  # uncomment if installed
)
model.eval()

# ── helper ─────────────────────────────────────────────────────────────────
def next_token_and_prob(prompt: str):
    """
    Return the model’s most-probable next token and its probability.
    """
    # Keep the tensor on the same device as the first model shard
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1,                # only need one step
            return_dict_in_generate=True,
            output_scores=True,               # scores for the new token
        )

    logits = out.scores[0][0]               # (vocab_size,)
    probs  = F.softmax(logits, dim=-1)

    top_idx   = int(torch.argmax(probs))
    top_prob  = float(probs[top_idx])
    top_token = tokenizer.convert_ids_to_tokens(top_idx)

    return top_token, top_prob

# ── main loop ───────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT)
df['prompt'] = df['prompt'].str.replace(r'\s*Answer:\s*$', '', regex=True) + 'The letter of the correct answer is **'

records = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="id"):
    top_tok, top_p = next_token_and_prob(row["prompt"])
    tqdm.write(f"{row['id']}: top_token='{top_tok}' ({top_p:.3f})")

    records.append({
        "id"            : row["id"],
        "subject"       : row["subject"],
        "question"      : row["prompt"],
        "correct"       : row["answer"].strip(),
        "top_token"     : top_tok,
        "top_token_prob": top_p,
    })

pd.DataFrame(records).to_csv(OUTPUT, index=False)
print(f"✅  Saved results ➜  {OUTPUT}")
