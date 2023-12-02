import pandas as pd
import joblib
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, f1_score
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings

RUN = 10

warnings.filterwarnings("ignore")

# Step 1: Load the test set
test_set = pd.read_csv('testset.csv')
row_indices = test_set['rowIndex']
X_test = test_set.drop('rowIndex', axis=1)
X_test2 = test_set.drop('rowIndex', axis=1)

# Load the training dataset
data = pd.read_csv('trainingset.csv')
row_indices_training = data['rowIndex']
X_train = data.iloc[:, 1:-1]  # Features
X_train = np.array(X_train)
X_train2 = data.iloc[:, 1:-1]  # Features
X_train2 = np.array(X_train2)
y_train = data['ClaimAmount']

# scale the classification data
scaler = joblib.load("sepehr_nn_classifier_scaler.joblib")
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


pca_classifier = joblib.load("sepehr_nn_classifier_pca.joblib")
X_train = pca_classifier.transform(X_train)
X_test = pca_classifier.transform(X_test)


# scale the regression data
scaler_nonzero = joblib.load('sepehr_nn_regressor_scaler.joblib')
X_train2 = scaler_nonzero.transform(X_train2)
X_test2 = scaler_nonzero.transform(X_test2)

pca = joblib.load('sepehr_nn_regressor_pca.joblib')
X_train2 = pca.transform(X_train2)
X_test2 = pca.transform(X_test2)

# rfe = joblib.load('sarah_best_reg_model_tuned_rfe.joblib')
# X_train2 = rfe.transform(X_train2)
# X_test2 = rfe.transform(X_test2)


# Step 2: Load the binary Keras model
# for .h5 models
binary_model = keras.models.load_model('sepehr_nn_classifier_model.h5', compile=False)
binary_model.compile()
binary_model_training = keras.models.load_model('sepehr_nn_classifier_model.h5', compile=False)
binary_model_training.compile()

# for .joblib models
# binary_model = joblib.load("sarah_best_model_tuned.joblib")
# binary_model_training = joblib.load("sarah_best_model_tuned.joblib")

# Step 3: Perform classification on the test set
binary_predictions_training = binary_model_training.predict(X_train)
binary_predictions = binary_model.predict(X_test)

pd_binary_pred = pd.DataFrame(binary_predictions)

# Initialize final_predictions_training outside the loop
final_predictions_training = pd.Series([0] * len(binary_predictions_training), index=row_indices_training)

# Create binary labels for classification (0 or 1)
binary_labels_training = (data['ClaimAmount'] != 0).astype(int)
binary_labels_test = (final_predictions_training != 0).astype(int)


# threshold is set to 0.5, which can be tuned in the future
binary_classes = binary_predictions
binary_classes_training = binary_predictions_training

# Analyze classification results
num_zeros = (binary_classes == 0).sum()
num_ones = (binary_classes == 1).sum()
total = len(binary_classes)
percent_zeros = (num_zeros / total) * 100
percent_ones = (num_ones / total) * 100

print(f"Class 0 (Zero): {num_zeros} samples ({percent_zeros:.2f}%)")
print(f"Class 1 (One): {num_ones} samples ({percent_ones:.2f}%)")

# Step 4: Load the sklearn regression model
# nonzero_model = joblib.load('sepehr_random_forest_nonzero_regressor.pkl') # run 1
# nonzero_model = joblib.load('gridsearch_nonzeroreg_5000outlier.joblib') # run 2
# nonzero_model = joblib.load('gridsearch_nonzeroreg_4646outlier.joblib') # run 3
# nonzero_model = joblib.load('gridsearch_nonzeroreg_20000outlier.joblib') # run 4
# nonzero_model = joblib.load("gridsearch_nonzeroreg_histgradboost_5000outlier.joblib") # run 5
# nonzero_model = joblib.load("sarah_best_reg_model_tuned.joblib") # run 6
nonzero_model = load_model("sepehr_nn_regressor_model.h5", compile=False)  # run 9 and 10
nonzero_model.compile()

# Step 5: Use the regression model to predict actual amounts for data points classified as 1
nonzero_indices = binary_classes.flatten() == 1
non_zero_indices_training = binary_classes_training.flatten() == 1
X_nonzero = X_test2[nonzero_indices]
X_nonzero_training = X_train2[non_zero_indices_training]

# Check if X_nonzero is empty
if X_nonzero.size == 0:
    print("No samples classified as non-zero. Skipping regression prediction.")
    nonzero_predictions = []
else:
    # Ensure the data format is consistent with the training format
    # You may need to adjust this depending on how your model was trained
    nonzero_predictions = nonzero_model.predict(X_nonzero)

if X_nonzero_training.size == 0:
    print("No samples classified as non-zero. Skipping regression prediction.")
    nonzero_predictions_training = []
else:
    # Ensure the data format is consistent with the training format
    # You may need to adjust this depending on how your model was trained
    nonzero_predictions_training = nonzero_model.predict(X_nonzero_training)

# Prepare the final predictions for saving
final_predictions = pd.Series([0] * len(binary_classes), index=row_indices)
final_predictions_training = pd.Series([0] * len(binary_classes_training), index=row_indices_training)

# Check if nonzero_predictions is not empty
if nonzero_predictions.size > 0:
    final_predictions.loc[row_indices[nonzero_indices]] = nonzero_predictions.flatten()

if nonzero_predictions_training.size > 0:
    final_predictions_training.loc[
        row_indices_training[non_zero_indices_training]] = nonzero_predictions_training.flatten()
    mae = mean_absolute_error(final_predictions_training, data['ClaimAmount'])
    print("Mae of training set: ", mae)

    f1 = f1_score(binary_labels_training, binary_classes_training, average='binary')
    print("F1 Score of training set: ", f1)

# Create a DataFrame for saving to CSV
predictions_df = pd.DataFrame({
    'rowIndex': final_predictions.index,
    'ClaimAmount': final_predictions.values
})

# Step 6: Save the predictions in a new CSV file
# checkpointnumber_groupnumber_submissionnumber.csv
predictions_df.to_csv(f'4_4_{RUN}.csv', index=False)
print(f"Predictions saved to '4_4_{RUN}.csv'")

