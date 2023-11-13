import pandas as pd
from sklearn.metrics import mean_absolute_error

# Load the first CSV file (actual values)
actual_data = pd.read_csv('training_submission_format.csv')  # Replace 'actual_values.csv' with the actual file name

# Load the second CSV file (predicted values)
predicted_data = pd.read_csv('output_results.csv')  # Replace 'submission_results.csv' with the actual file name

# Merge the two DataFrames on 'rowIndex' to ensure the order is the same
merged_data = pd.merge(actual_data, predicted_data, on='rowIndex', suffixes=('_actual', '_predicted'))

# Calculate MAE
mae = mean_absolute_error(merged_data['ClaimAmount_actual'], merged_data['ClaimAmount_predicted'])
print(f'MAE: {mae}')
