import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('trainingset.csv')

# Separate features and labels
X = data.iloc[:, 1:-1]  # Features
data['ClaimLabel'] = (data['ClaimAmount'] > 0).astype(int)
y = data['ClaimLabel']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Feature Selection with RFE
rf_classifier_for_rfe = RandomForestClassifier(random_state=42)
rfe = RFE(estimator=rf_classifier_for_rfe, n_features_to_select=10, verbose=2)  # Adjust the number of features
X_train = rfe.fit_transform(X_train, y_train)
X_test = rfe.transform(X_test)

# Dimensionality Reduction with PCA
pca = PCA(n_components=0.95)  # Adjust the number of components
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# 4. Hyperparameter Tuning
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 15, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, 
                           cv=5, 
                           verbose=2, 
                           n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Find the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Evaluate the best model
best_rf = grid_search.best_estimator_
test_accuracy = best_rf.score(X_test, y_test)
print(f"Test Accuracy of Best Model: {test_accuracy}")

# Get the best model from grid search
best_rf = grid_search.best_estimator_

# Predictions on training set
y_train_pred = best_rf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Predictions on test set
y_test_pred = best_rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

# Print the results
print(f"Training Set Accuracy: {train_accuracy}")
print(f"Training Set F1 Score: {train_f1}")
print(f"Test Set Accuracy: {test_accuracy}")
print(f"Test Set F1 Score: {test_f1}")

# # Convert X_train_balanced (numpy.ndarray) to a DataFrame
# X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=X_test.columns)

# Combine the training and test sets
X_full = np.concatenate([X_train, X_test])
y_full = pd.concat([y_train, y_test])

# Train the best model on the entire dataset
best_rf.fit(X_full, y_full)

# Save the retrained model
dump(best_rf, 'reduced_feature_random_forest_model.joblib')