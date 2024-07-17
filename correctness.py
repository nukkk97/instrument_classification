import pandas as pd

predictions = pd.read_csv('predictions.csv', header=None)
answers = pd.read_csv('Metadata_Test.csv', header=None)

predictions.columns = ['question', 'predicted_answer']
answers.columns = ['question', 'correct_answer']

merged_df = pd.merge(predictions, answers, on='question', how='inner')

correct_predictions = (merged_df['predicted_answer'] == merged_df['correct_answer']).sum()

total_predictions = len(merged_df)
correctness_ratio = correct_predictions / total_predictions

print(f"Correctness Ratio: {correctness_ratio:.2f}")
