#!/usr/bin/env python3
"""
Generate reasoning processes for MMLU prompts using gpt-oss models.
Input: mmlu_prompts_gpt_train.csv and mmlu_prompts_gpt_test.csv
Output: CSV files with added reasoning column (prompt_r)
"""

import csv
import pathlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re

# Configuration
MODEL_NAME = "openai/gpt-oss-20b"  # or "openai/gpt-oss-120b" for larger model
TRAIN_INFILE = pathlib.Path("mmlu_prompts_gpt_train.csv")
TEST_INFILE = pathlib.Path("mmlu_prompts_gpt_test.csv")
TRAIN_OUTFILE = pathlib.Path("mmlu_prompts_reasoning_gpt20_train.csv")
TEST_OUTFILE = pathlib.Path("mmlu_prompts_reasoning_gpt20_test.csv")

# Batch processing parameters
BATCH_SIZE = 16  # Adjust based on your GPU memory
MAX_NEW_TOKENS = 512  # Max tokens for reasoning generation
TEMPERATURE = 0.7  # Temperature for generation

def load_model_and_tokenizer(model_name):
    """Load the gpt-oss model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"  # Automatically place on available GPUs
    )
    return model, tokenizer

def extract_reasoning_from_output(generated_text, original_prompt):
    """
    Extract the reasoning process from the model's output.
    The reasoning appears in the analysis channel.
    """
    # Remove the original prompt from the generated text
    if original_prompt in generated_text:
        generated_text = generated_text.replace(original_prompt, "").strip()
    
    # Extract content from analysis channel
    # Pattern to match analysis channel content
    analysis_pattern = r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|start\|>)'
    
    reasoning_parts = []
    for match in re.finditer(analysis_pattern, generated_text, re.DOTALL):
        reasoning_parts.append(match.group(1).strip())
    
    # If no analysis channel found, try to extract any reasoning before final channel
    if not reasoning_parts:
        # Try to extract content before final channel
        final_pattern = r'^(.*?)(?:<\|channel\|>final|<\|start\|>assistant<\|channel\|>final)'
        match = re.search(final_pattern, generated_text, re.DOTALL)
        if match and match.group(1).strip():
            reasoning_parts.append(match.group(1).strip())
    
    # Join all reasoning parts
    reasoning = " ".join(reasoning_parts)
    
    # Clean up any remaining special tokens
    reasoning = re.sub(r'<\|[^|]+\|>', '', reasoning).strip()
    
    return reasoning if reasoning else "Let me analyze this question step by step."

def generate_reasoning_batch(model, tokenizer, prompts, batch_size=4):
    """
    Generate reasoning for a batch of prompts.
    Returns a list of reasoning texts.
    """
    reasoning_list = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Prepare inputs for batch processing
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Generate with reasoning enabled
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                # Stop at these tokens
                eos_token_id=[
                    tokenizer.convert_tokens_to_ids(["<|return|>"])[0] if "<|return|>" in tokenizer.get_vocab() else tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids(["<|call|>"])[0] if "<|call|>" in tokenizer.get_vocab() else tokenizer.eos_token_id,
                ],
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode outputs
        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=False)
            original_prompt = batch_prompts[j % len(batch_prompts)]
            reasoning = extract_reasoning_from_output(generated_text, original_prompt)
            reasoning_list.append(reasoning)
    
    return reasoning_list

def modify_prompt_for_reasoning(prompt):
    """
    Modify the prompt to enable high-level reasoning.
    Insert system message with reasoning: high before the prompt.
    """
    # Check if prompt already has system message
    if "<|start|>system" in prompt:
        # Modify existing system message to include reasoning: high
        prompt = re.sub(
            r'(<\|start\|>system<\|message\|>.*?)((?:Reasoning:\s*\w+\s*)?)(.*?<\|end\|>)',
            r'\1Reasoning: high\n\n\3',
            prompt,
            flags=re.DOTALL
        )
    else:
        # Add system message with reasoning at the beginning
        system_message = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-06-28

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>
"""
        prompt = system_message + prompt
    
    # Ensure the prompt ends properly for assistant generation
    if not prompt.endswith("<|start|>assistant"):
        if prompt.endswith("**"):
            # Remove the trailing ** and add proper assistant start
            prompt = prompt[:-2] + "<|end|>\n<|start|>assistant"
        else:
            prompt += "\n<|start|>assistant"
    
    return prompt

def create_prompt_with_reasoning(original_prompt, reasoning):
    """
    Create the final prompt_r column value.
    Format: original_prompt + reasoning + 'The letter of the correct answer is **'
    """
    # Remove the trailing part from original prompt if it exists
    if original_prompt.endswith('The letter of the correct answer is **'):
        base_prompt = original_prompt[:-len('The letter of the correct answer is **')].rstrip()
    else:
        base_prompt = original_prompt.rstrip()
    
    # Construct the final prompt with reasoning
    prompt_r = f"{base_prompt}\n\n{reasoning}\n\nThe letter of the correct answer is **"
    
    return prompt_r

def process_csv_file(infile, outfile, model, tokenizer):
    """Process a single CSV file and add reasoning column."""
    print(f"\nProcessing {infile}")
    
    # Read all rows
    rows = []
    with infile.open(encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
    
    print(f"Found {len(rows)} rows to process")
    
    # Prepare prompts for reasoning generation
    modified_prompts = []
    for row in rows:
        modified_prompt = modify_prompt_for_reasoning(row["prompt"])
        modified_prompts.append(modified_prompt)
    
    # Generate reasoning in batches
    print("Generating reasoning...")
    reasoning_list = []
    for i in tqdm(range(0, len(modified_prompts), BATCH_SIZE)):
        batch = modified_prompts[i:i+BATCH_SIZE]
        batch_reasoning = generate_reasoning_batch(model, tokenizer, batch, batch_size=1)
        reasoning_list.extend(batch_reasoning)
    
    # Create new rows with reasoning
    new_rows = []
    for i, row in enumerate(rows):
        new_row = row.copy()
        # Create prompt_r with reasoning
        new_row["prompt_r"] = create_prompt_with_reasoning(row["prompt"], reasoning_list[i])
        new_rows.append(new_row)
    
    # Write output CSV
    print(f"Writing to {outfile}")
    with outfile.open("w", encoding="utf-8", newline="") as fout:
        fieldnames = ["id", "subject", "prompt", "prompt_r", "answer"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)
    
    print(f"Successfully processed {len(new_rows)} rows")
    
    # Print a sample for verification
    if new_rows:
        print("\n--- Sample Output ---")
        print(f"ID: {new_rows[0]['id']}")
        print(f"Subject: {new_rows[0]['subject']}")
        print(f"Answer: {new_rows[0]['answer']}")
        print(f"Prompt_r preview (first 500 chars):")
        print(new_rows[0]['prompt_r'][:500])
        print("...")

def main():
    """Main processing function."""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    # Process train file if it exists
    if TRAIN_INFILE.exists():
        process_csv_file(TRAIN_INFILE, TRAIN_OUTFILE, model, tokenizer)
    else:
        print(f"Warning: {TRAIN_INFILE} not found, skipping...")
    
    # Process test file if it exists
    if TEST_INFILE.exists():
        process_csv_file(TEST_INFILE, TEST_OUTFILE, model, tokenizer)
    else:
        print(f"Warning: {TEST_INFILE} not found, skipping...")
    
    print("\n✅ Processing complete!")
    if TRAIN_OUTFILE.exists():
        print(f"Train output: {TRAIN_OUTFILE.resolve()}")
    if TEST_OUTFILE.exists():
        print(f"Test output: {TEST_OUTFILE.resolve()}")

if __name__ == "__main__":
    main()