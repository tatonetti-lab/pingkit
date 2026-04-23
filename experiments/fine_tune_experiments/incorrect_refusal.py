import pandas as pd

# Read both CSV files
medmcqa_refused = pd.read_csv('medmcqa_eval_refused.csv')
gemma_incorrect = pd.read_csv('gemma_medmcqa_top_token_incorrect_only.csv')

# Keep only the columns needed from the second file
gemma_top_token = gemma_incorrect[['id', 'top_token']]

# Filter medmcqa_refused to matching IDs and add top_token
filtered_data = medmcqa_refused.merge(gemma_top_token, on='id', how='inner')

# Save the filtered data, now including top_token
filtered_data.to_csv('medmcqa_eval_refused_incorrect.csv', index=False)

print(f"Original rows: {len(medmcqa_refused)}")
print(f"Valid IDs: {gemma_incorrect['id'].nunique()}")
print(f"Filtered rows: {len(filtered_data)}")
print("Filtered data saved to 'medmcqa_eval_refused_incorrect.csv'")