import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── settings ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-9b-it"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
INPUT      = "mmlu_prompts_gemma_test.csv"
OUTPUT     = "mmlu_gemma_9b_test_COT.csv"

# ── load model ────────────────────────────────────────────────────────────────
print("Loading model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ── helper ────────────────────────────────────────────────────────────────────
def next_token_and_prob(prompt: str):
    """
    Returns the model’s single most-probable next token and its probability.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    out = model.generate(
        **inputs,
        max_new_tokens=1,               # only care about the first token
        return_dict_in_generate=True,
        output_scores=True
    )

    logits = out.scores[0][0]          # (vocab_size,)
    probs  = F.softmax(logits, dim=-1) # still on DEVICE

    top_idx   = int(torch.argmax(probs))
    top_prob  = float(probs[top_idx])
    top_token = tokenizer.convert_ids_to_tokens(top_idx)

    return top_token, top_prob

# ── main loop ─────────────────────────────────────────────────────────────────
df       = pd.read_csv(INPUT)
df['prompt'] = df['prompt'].str.replace(r'\s*Answer:\s*$', '', regex=True) + 'Letter of the correct answer **'

records  = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="id"):
    top_tok, top_p = next_token_and_prob(row["prompt"])
    tqdm.write(f"{row['id']}: top_token='{top_tok}' ({top_p:.3f})")

    records.append({
        "id"            : row["id"],
        "subject"       : row["subject"],
        "question"      : row["prompt"],
        "correct"       : row["answer"].strip(),   # keep if you still want it
        "top_token"     : top_tok,
        "top_token_prob": top_p,
    })

pd.DataFrame(records).to_csv(OUTPUT, index=False)
print(f"Saved results ➜  {OUTPUT}")
