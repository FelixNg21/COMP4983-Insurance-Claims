import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
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
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Instantiate the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree Model
decision_tree.fit(X_train_balanced, y_train_balanced)

# Evaluate the Model
train_accuracy = decision_tree.score(X_train_balanced, y_train_balanced)
test_accuracy = decision_tree.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Hyperparameter Tuning
# Define the parameter grid for the decision tree
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), 
                           param_grid=param_grid, 
                           cv=5, 
                           verbose=2, 
                           n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train_balanced, y_train_balanced)

# Find the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Evaluate the best model
best_decision_tree = grid_search.best_estimator_
test_accuracy = best_decision_tree.score(X_test, y_test)
print(f"Test Accuracy of Best Model: {test_accuracy}")

# Predictions on training set
y_train_pred = best_decision_tree.predict(X_train_balanced)
train_accuracy = accuracy_score(y_train_balanced, y_train_pred)
train_f1 = f1_score(y_train_balanced, y_train_pred)

# Predictions on test set
y_test_pred = best_decision_tree.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

# Print the results
print(f"Training Set Accuracy: {train_accuracy}")
print(f"Training Set F1 Score: {train_f1}")
print(f"Test Set Accuracy: {test_accuracy}")
print(f"Test Set F1 Score: {test_f1}")

# Combine the training and test sets
X_full = np.concatenate([X_train_balanced, X_test])
y_full = pd.concat([y_train_balanced, y_test])

# Train the best model on the entire dataset
best_decision_tree.fit(X_full, y_full)

# Save the retrained model
dump(best_decision_tree, 'decision_tree_model.joblib')