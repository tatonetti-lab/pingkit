"""
Build the labeled SimpleQA correctness dataset from GPT-judged generations.

Unlike split_boolq.py (which makes a balanced 50/50 BoolQ split), this script
preserves SimpleQA's natural class imbalance: it labels every scored question
correct/incorrect and writes a single file. The train/holdout partition is done
downstream by train_correctness.py, which draws a fixed 300 incorrect / 200
correct training set and holds out the (imbalanced) remainder for evaluation.

Input:  simpleqa_scored.csv  (columns: id, metadata, problem, true_answer,
        answer, score, reasoning) where `score` is CORRECT / INCORRECT / ERROR.
Output: simpleqa_labeled.csv (id, problem, true_answer, answer, correct, prompt).
"""
import pandas as pd

INPUT = "simpleqa_scored.csv"
OUTPUT = "simpleqa_labeled.csv"

# Gemma prompt used to elicit the generated answer; the answer is appended so the
# probe reads the activation at the last token of the model's own response.
PROMPT_TEMPLATE = """<start_of_turn>user
Answer the following question. Reply with a single phrase that just contains the answer, do not use a complete sentence.

{question}<end_of_turn>
<start_of_turn>model
{answer}"""


def ensure_period(text: str) -> str:
    text = str(text).strip()
    return text if text.endswith(".") else text + "."


def format_prompt(question: str, answer: str) -> str:
    return PROMPT_TEMPLATE.format(question=question, answer=ensure_period(answer))


def main() -> None:
    print(f"Loading {INPUT}...")
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} rows")

    # Drop questions the judge could not score (Azure content-policy rejections).
    error_count = int((df["score"] == "ERROR").sum())
    if error_count:
        print(f"Removing {error_count} ERROR rows (unscored)")
        df = df[df["score"] != "ERROR"].reset_index(drop=True)

    # Binary label: 1 = correct, 0 = everything else (INCORRECT / INC_toggle).
    df["correct"] = (df["score"] == "CORRECT").astype(int)
    n_correct = int(df["correct"].sum())
    n_incorrect = len(df) - n_correct
    print(f"CORRECT: {n_correct}, INCORRECT: {n_incorrect} "
          f"({100 * n_correct / len(df):.1f}% correct)")

    df["prompt"] = df.apply(
        lambda row: format_prompt(row["problem"], row["answer"]), axis=1
    )

    out = df[["id", "problem", "true_answer", "answer", "correct", "prompt"]]
    out.to_csv(OUTPUT, index=False)
    print(f"Saved {len(out)} labeled rows -> {OUTPUT}")
    print("The held-out evaluation set retains this imbalance; see run_simpleqa_eval.sh.")


if __name__ == "__main__":
    main()
