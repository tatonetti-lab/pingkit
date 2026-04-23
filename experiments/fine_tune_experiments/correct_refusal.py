import pandas as pd

# Read both CSV files
medmcqa_refused = pd.read_csv('medmcqa_eval_refused.csv')
gemma_correct = pd.read_csv('gemma_medmcqa_top_token_correct_only.csv')

# Get the set of IDs from the second file
valid_ids = set(gemma_correct['id'])

# Filter the first CSV to only keep rows with IDs that exist in the second CSV
filtered_data = medmcqa_refused[medmcqa_refused['id'].isin(valid_ids)]

# Save the filtered data, preserving the original structure
filtered_data.to_csv('medmcqa_eval_refused_filtered.csv', index=False)

print(f"Original rows: {len(medmcqa_refused)}")
print(f"Valid IDs: {len(valid_ids)}")
print(f"Filtered rows: {len(filtered_data)}")
print("Filtered data saved to 'medmcqa_eval_refused_correct.csv'")
