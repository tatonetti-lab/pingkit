import pandas as pd
import os
from pingkit import embed_dataset
from transformers import AutoTokenizer



train_df = pd.read_csv('medmcqa_eval_correct.csv')
test_df= pd.read_csv('medmcqa_eval_refused.csv')

set_base = "MedMCQA"
model="google/gemma-2-9b-it"
model_base=model.split('/')[-1]
train_dir=set_base+"_"+model_base+"_correct"
test_dir=set_base+"_"+model_base+"_refused"
text_col='prompt'

# Build Gemma chat-formatted inputs and append "Answer: " at the start of the assistant turn.
tokenizer = AutoTokenizer.from_pretrained(model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def to_gemma_input(user_prompt: str, prefix: str = "Answer:") -> str:
    # Do NOT modify the original prompt; wrap it with the chat template and open the assistant turn.
    messages = [{"role": "user", "content": str(user_prompt)}]
    s = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Seed the assistant with the requested prefix
    if not s.endswith(prefix):
        s = s + prefix
    return s

# Create a new column with the exact text to embed
train_df = train_df.copy()
test_df = test_df.copy()
train_df["gemma_input"] = train_df[text_col].astype(str).apply(to_gemma_input)
test_df["gemma_input"] = test_df[text_col].astype(str).apply(to_gemma_input)

# (Optional) quick peek at one example
print("\n[DEBUG] Sample train gemma_input:\n", train_df["gemma_input"].iloc[0][:500])

# Correct set
embed_dataset(
    train_df,
    input_col="gemma_input",
    output_dir=train_dir,
    model_name=model,
    lora_adapter="model",
    device="auto",
    pooling="last",
    filter_non_text=False,
)

# Refused set
embed_dataset(
    test_df,
    input_col="gemma_input",
    output_dir=test_dir,
    model_name=model,
    lora_adapter="model",
    device="auto",
    pooling="last",
    filter_non_text=False,
)
