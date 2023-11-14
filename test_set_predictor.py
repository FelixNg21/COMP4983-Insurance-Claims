import pandas as pd
import joblib
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 1: Load the test set
test_set = pd.read_csv('testset.csv')
row_indices = test_set['rowIndex']
X_test = test_set.drop('rowIndex', axis=1)


# Load the training dataset
data = pd.read_csv('trainingset.csv')
X_train = data.iloc[:, 1:-1]  # Features
X_train = np.array(X_train)
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Step 2: Load the binary Keras model
binary_model = keras.models.load_model('trained_model.h5')

# Step 3: Perform classification on the test set
binary_predictions = binary_model.predict(X_test)
binary_classes = (binary_predictions > 0.5).astype("int32")

# Analyze classification results
num_zeros = (binary_classes == 0).sum()
num_ones = (binary_classes == 1).sum()
total = len(binary_classes)
percent_zeros = (num_zeros / total) * 100
percent_ones = (num_ones / total) * 100

print(f"Class 0 (Zero): {num_zeros} samples ({percent_zeros:.2f}%)")
print(f"Class 1 (One): {num_ones} samples ({percent_ones:.2f}%)")


# Step 4: Load the sklearn regression model
nonzero_model = joblib.load('trained_model_nonzero.pkl')

# Step 5: Use the regression model to predict actual amounts for data points classified as 1
nonzero_indices = binary_classes.flatten() == 1
X_nonzero = X_test[nonzero_indices]

# Check if X_nonzero is empty
if X_nonzero.size == 0:
    print("No samples classified as non-zero. Skipping regression prediction.")
    nonzero_predictions = []
else:
    # Ensure the data format is consistent with the training format
    # You may need to adjust this depending on how your model was trained
    nonzero_predictions = nonzero_model.predict(X_nonzero)

# Prepare the final predictions for saving
final_predictions = pd.Series([0] * len(binary_classes), index=row_indices)

# Check if nonzero_predictions is not empty
if nonzero_predictions.size > 0:
    final_predictions.loc[row_indices[nonzero_indices]] = nonzero_predictions.flatten()

# Create a DataFrame for saving to CSV
predictions_df = pd.DataFrame({
    'rowIndex': final_predictions.index,
    'claimAmount': final_predictions.values
})

# Step 6: Save the predictions in a new CSV file
predictions_df.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")
