import pandas as pd
import json
from openai import AzureOpenAI
from tqdm import tqdm

# ── settings ──────────────────────────────────────────────────────────────────
MODEL_NAME = "gpt-5.2"
API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_KEY = ""
AZURE_OPENAI_ENDPOINT = ""


INPUT  = "simpleqa_results.csv"
OUTPUT = "simpleqa_scored.csv"

# ── initialize Azure client ───────────────────────────────────────────────────
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ── scoring prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert judge evaluating whether a model's answer to a question is correct.

Your task:
1. You will be given a QUESTION, the TRUE ANSWER, and the MODEL'S ANSWER.
2. Determine if the model's answer is correct or incorrect.
3. Be flexible with minor differences in formatting, phrasing, or punctuation.
4. Consider an answer correct if it conveys the same information as the true answer, even if worded differently.

Scoring criteria:
- CORRECT: The model's answer matches the true answer in meaning (even if phrased differently)
- INCORRECT: The model's answer is wrong or does not match the true answer

Respond with JSON in this exact format:
{"score": "CORRECT" or "INCORRECT", "reasoning": "brief explanation of your judgment"}"""

# ── helper ────────────────────────────────────────────────────────────────────
def score_answer(question: str, true_answer: str, model_answer: str) -> dict:
    """Score the model's answer against the true answer using GPT-5."""
    
    user_prompt = f"""Question: {question}

True Answer: {true_answer}

Model's Answer: {model_answer}

Is the model's answer correct?"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,  # deterministic scoring
            response_format={"type": "json_object"},
            max_completion_tokens=256
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "score": result.get("score", "ERROR"),
            "reasoning": result.get("reasoning", "")
        }
    
    except Exception as e:
        print(f"Error scoring answer: {e}")
        return {"score": "ERROR", "reasoning": str(e)}

# ── main loop ─────────────────────────────────────────────────────────────────
print(f"Loading data from {INPUT}…")
df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} rows")

records = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
    result = score_answer(
        question=row["problem"],
        true_answer=row["true_answer"],
        model_answer=row["answer"]
    )
    
    tqdm.write(f"{row['id']}: {result['score']}")
    
    records.append({
        "id":          row["id"],
        "metadata":    row["metadata"],
        "problem":     row["problem"],
        "true_answer": row["true_answer"],
        "answer":      row["answer"],
        "score":       result["score"],
        "reasoning":   result["reasoning"],
    })

result_df = pd.DataFrame(records)
result_df.to_csv(OUTPUT, index=False)
print(f"\nSaved {len(result_df)} results → {OUTPUT}")

# ── summary statistics ────────────────────────────────────────────────────────
print("\n" + "="*50)
print("SCORING SUMMARY")
print("="*50)
score_counts = result_df["score"].value_counts()
total = len(result_df)

for score_type in ["CORRECT", "INCORRECT", "ERROR"]:
    count = score_counts.get(score_type, 0)
    pct = (count / total) * 100
    print(f"{score_type}: {count} ({pct:.1f}%)")

accuracy = score_counts.get("CORRECT", 0) / total * 100
print(f"\nAccuracy: {accuracy:.1f}%")

# ── preview ───────────────────────────────────────────────────────────────────
print("\n--- Sample CORRECT ---")
correct_samples = result_df[result_df["score"] == "CORRECT"]
if len(correct_samples) > 0:
    sample = correct_samples.iloc[0]
    print(f"Problem: {sample['problem']}")
    print(f"True: {sample['true_answer']}")
    print(f"Model: {sample['answer']}")
    print(f"Reasoning: {sample['reasoning']}")

print("\n--- Sample INCORRECT ---")
incorrect_samples = result_df[result_df["score"] == "INCORRECT"]
if len(incorrect_samples) > 0:
    sample = incorrect_samples.iloc[0]
    print(f"Problem: {sample['problem']}")
    print(f"True: {sample['true_answer']}")
    print(f"Model: {sample['answer']}")
    print(f"Reasoning: {sample['reasoning']}")