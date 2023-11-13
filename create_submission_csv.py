import pandas as pd

# Load the new test dataset
new_data = pd.read_csv('trainingset.csv')  # Replace 'new_test_set.csv' with the actual file name

# Create a DataFrame in the submission format
submission_df = pd.DataFrame({
    'rowIndex': new_data['rowIndex'],
    'ClaimAmount': new_data['ClaimAmount']
})

# Save the submission DataFrame to a new CSV file
submission_df.to_csv('training_submission_format.csv', index=False)

# Create a DataFrame with all 'ClaimAmount' values set to zeros
all_zeros_df = pd.DataFrame({
    'rowIndex': new_data['rowIndex'],
    'ClaimAmount': 0
})

# Save the DataFrame with zeros to a new CSV file
all_zeros_df.to_csv('all_zeros.csv', index=False)
