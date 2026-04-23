import pandas as pd

# ── settings ──────────────────────────────────────────────────────────────────
INPUT  = "boolq_scored.csv"
OUTPUT_TRAIN = "boolq_train.csv"
OUTPUT_TEST  = "boolq_test.csv"
SEED = 42  # random seed for reproducibility

# ── load data ─────────────────────────────────────────────────────────────────
print(f"Loading data from {INPUT}…")
df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} rows")

# ── filter out errors ─────────────────────────────────────────────────────────
error_count = (df["score"] == "ERROR").sum()
if error_count > 0:
    print(f"Removing {error_count} ERROR rows")
    df = df[df["score"] != "ERROR"].reset_index(drop=True)
    print(f"Remaining: {len(df)} rows")

# ── convert score to bool ─────────────────────────────────────────────────────
df["correct"] = (df["score"] == "CORRECT").astype(int)
total_correct = int(df["correct"].sum())
total_incorrect = int(len(df) - total_correct)
print(f"CORRECT: {total_correct}, INCORRECT: {total_incorrect}")

# ── helper to ensure answer ends with period ──────────────────────────────────
def ensure_period(text: str) -> str:
    """Ensure the text ends with a period."""
    text = str(text).strip()
    if not text.endswith("."):
        text = text + "."
    return text

# ── format prompt ─────────────────────────────────────────────────────────────
def format_prompt(question: str, answer: str) -> str:
    """Format as Gemma conversation with answer included."""
    answer = ensure_period(answer)

    prompt = f"""<start_of_turn>user
Answer the following question in a single phrase:

{question}<end_of_turn>
<start_of_turn>model
{answer}"""

    return prompt

# ── create prompt column ──────────────────────────────────────────────────────
print("Formatting prompts…")
df["prompt"] = df.apply(lambda row: format_prompt(row["question"], row["answer"]), axis=1)

# ── balanced 50/50 train/test split (each set has equal class counts) ─────────
print("\nCreating balanced 50/50 train/test split…")
correct_df = df[df["correct"] == 1]
incorrect_df = df[df["correct"] == 0]

min_class = min(len(correct_df), len(incorrect_df))
# Ensure we can split each class evenly into train/test halves
n_per_class_total = (min_class // 2) * 2

if n_per_class_total == 0:
    raise ValueError(
        f"Not enough data to make a balanced 50/50 split. "
        f"Correct={len(correct_df)}, Incorrect={len(incorrect_df)}"
    )

n_per_class_each_split = n_per_class_total // 2

dropped_correct = len(correct_df) - n_per_class_total
dropped_incorrect = len(incorrect_df) - n_per_class_total
print(f"Using {n_per_class_total} examples per class (downsampling to minority class; even for exact 50/50 split).")
print(f"Dropping (unused): Correct={dropped_correct}, Incorrect={dropped_incorrect}")

# Sample equal counts from each class
correct_sampled = correct_df.sample(n=n_per_class_total, random_state=SEED).reset_index(drop=True)
incorrect_sampled = incorrect_df.sample(n=n_per_class_total, random_state=SEED).reset_index(drop=True)

# Split each class exactly 50/50
correct_train = correct_sampled.iloc[:n_per_class_each_split].copy()
correct_test  = correct_sampled.iloc[n_per_class_each_split:].copy()

incorrect_train = incorrect_sampled.iloc[:n_per_class_each_split].copy()
incorrect_test  = incorrect_sampled.iloc[n_per_class_each_split:].copy()

print(f"Train: Correct={len(correct_train)}, Incorrect={len(incorrect_train)}, Total={len(correct_train) + len(incorrect_train)}")
print(f"Test:  Correct={len(correct_test)},  Incorrect={len(incorrect_test)},  Total={len(correct_test) + len(incorrect_test)}")

# ── combine into train/test sets ──────────────────────────────────────────────
train_df = (
    pd.concat([correct_train, incorrect_train], ignore_index=True)
      .sample(frac=1, random_state=SEED)
      .reset_index(drop=True)
)
test_df = (
    pd.concat([correct_test, incorrect_test], ignore_index=True)
      .sample(frac=1, random_state=SEED)
      .reset_index(drop=True)
)

# ── select output columns ─────────────────────────────────────────────────────
output_columns = ["id", "question", "true_answer", "answer", "perplexity","correct", "prompt"]
train_df = train_df[output_columns]
test_df = test_df[output_columns]

# ── save ─────────────────────────────────────────────────────────────────────
train_df.to_csv(OUTPUT_TRAIN, index=False)
test_df.to_csv(OUTPUT_TEST, index=False)

print(f"\nSaved {len(train_df)} rows → {OUTPUT_TRAIN}")
print(f"Saved {len(test_df)} rows → {OUTPUT_TEST}")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("SPLIT SUMMARY")
print("="*50)

print(f"\nTrain set (BALANCED):")
print(f"  Total: {len(train_df)}")
print(f"  Correct: {train_df['correct'].sum()} ({train_df['correct'].mean()*100:.1f}%)")
print(f"  Incorrect: {len(train_df) - train_df['correct'].sum()} ({(1-train_df['correct'].mean())*100:.1f}%)")

print(f"\nTest set (BALANCED):")
print(f"  Total: {len(test_df)}")
print(f"  Correct: {test_df['correct'].sum()} ({test_df['correct'].mean()*100:.1f}%)")
print(f"  Incorrect: {len(test_df) - test_df['correct'].sum()} ({(1-test_df['correct'].mean())*100:.1f}%)")

# ── preview ───────────────────────────────────────────────────────────────────
print("\n--- Sample CORRECT prompt ---")
correct_sample = train_df[train_df["correct"] == 1].iloc[0]
print(f"ID: {correct_sample['id']}")
print(f"Correct: {correct_sample['correct']}")
print(f"Prompt:\n{correct_sample['prompt']}")

print("\n--- Sample INCORRECT prompt ---")
incorrect_sample = train_df[train_df["correct"] == 0].iloc[0]
print(f"ID: {incorrect_sample['id']}")
print(f"Correct: {incorrect_sample['correct']}")
print(f"Prompt:\n{incorrect_sample['prompt']}")
