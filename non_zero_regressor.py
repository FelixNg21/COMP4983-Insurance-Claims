import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
import joblib  # For saving the model

OUTLIER = 5000

# def evaluate_model_with_cross_validation(model, X, y, n_splits=5):
#     mse_scores = cross_val_score(model, X, y, cv=n_splits, scoring='neg_mean_squared_error')
#     mae_scores = cross_val_score(model, X, y, cv=n_splits, scoring='neg_mean_absolute_error')
#
#     mean_mse = np.mean(mse_scores)
#     mean_mae = np.mean(mae_scores)
#
#     return -mean_mse, -mean_mae


# Load and prepare the dataset
data = pd.read_csv('trainingset.csv')
non_zero_data = data[data['ClaimAmount'] > 0]
X_non_zero_with_outlier = non_zero_data.iloc[:, 1:-1]
y_non_zero_with_outlier = non_zero_data['ClaimAmount']

#######
# Check if we need to remove outliers!!!
# changed outlier from 5000 to 4646.23 for run 2 (gridsearch_nonzeroreg_4646outlier.joblib)
non_zero_data_no_outlier = non_zero_data[non_zero_data['ClaimAmount'] < OUTLIER]
X_non_zero_no_outlier = non_zero_data_no_outlier.iloc[:, 1:-1]
y_non_zero_no_outlier = non_zero_data_no_outlier['ClaimAmount']

# Standardize features
scaler = StandardScaler()
X_non_zero_no_outlier_scaled = scaler.fit_transform(X_non_zero_no_outlier)


# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# # added in random state run 2 (gridsearch_nonzeroreg_4646outlier.joblib)
# RFR = RandomForestRegressor(random_state=42)

# param_grid = { # for histogram gradient boosting
#     'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
#     'max_iter': [50, 100, 150],  # Number of boosting iterations
#     'max_depth': [3, 4, 5],  # Maximum depth of the individual trees
#     'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
#     'max_bins': [100, 150, 200, 255],  # Maximum number of bins for histogram splitting
#     'l2_regularization': [0.0, 0.1, 0.2],  # L2 regularization term on weights
#     'max_leaf_nodes': [None, 31, 63],  # Maximum number of leaves for each tree
#     'random_state': [42]  # Set to a specific value for reproducibility
# }
#
# RFR = HistGradientBoostingRegressor(random_state=42)

param_grid = { # for gradient boosting
    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],  # Loss function to be optimized
    'n_estimators': [50, 100, 150],  # Number of boosting stages to be run
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
    'criterion': ['friedman_mse', 'squared_error'],  # Function to measure the quality of a split
    'max_depth': [3, 4, 5, 6, None],  # Maximum depth of the individual trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'subsample': [0.8, 1.0],  # Fraction of samples used for fitting the trees
    'random_state': [42]  # Set to a specific value for reproducibility
    # 'max_features': ['sqrt', 'log2', None]  # Number of features to consider for the best split
}

RFR = GradientBoostingRegressor()

# changed cv to 5 for run 2 (gridsearch_nonzeroreg_4646outlier.joblib)
grid_nonzero = GridSearchCV(estimator=RFR, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1, scoring='neg_mean_absolute_error')
grid_nonzero.fit(X_non_zero_no_outlier_scaled, y_non_zero_no_outlier)

print("Best Hyperparameters:", grid_nonzero.best_params_)
# Use the best model to make predictions on the test set
best_model = grid_nonzero.best_estimator_
print("Best Model:", best_model)
y_pred = best_model.predict(X_non_zero_no_outlier_scaled)
pd.DataFrame(y_pred).to_csv('sarah_y_pred_nzr.csv')
pd.DataFrame(y_non_zero_no_outlier).to_csv('sarah_y_test_nzr.csv')

best_score = grid_nonzero.best_score_
print("Best Score:", best_score)


X_all = scaler.fit_transform(X_non_zero_with_outlier, y_non_zero_with_outlier)
best_model.fit(X_all, y_non_zero_with_outlier)
joblib.dump(scaler, f'gridsearch_nonzeroreg_gradboost_{OUTLIER}outlier_scaler.joblib')
joblib.dump(best_model, f'gridsearch_nonzeroreg_gradboost_{OUTLIER}outlier.joblib')

# # Define models to evaluate
# models = [
#     Lasso(alpha=0.001),
#     Ridge(alpha=0.001),
#     RandomForestRegressor(),
#     GradientBoostingRegressor(),
#     SVR(),
#     HuberRegressor()
# ]
#
# # Evaluate each model and track their performance
# model_performance = {}
# for model in models:
#     mse, mae = evaluate_model_with_cross_validation(model, X_non_zero_no_outlier_scaled, y_non_zero)
#     model_performance[model.__class__.__name__] = (mse, mae)

# Identify the best model (lowest MAE)
# best_model_name = min(model_performance, key=lambda k: model_performance[k][1])
# print(best_model_name)
# best_model = [model for model in models if model.__class__.__name__ == best_model_name][0]
#
# # Fit the best model on the entire dataset
# best_model.fit(X_non_zero_no_outlier_scaled, y_non_zero)
#
#
# # Step 1: Save the model
# joblib.dump(best_model, 'trained_model_nonzero.pkl')
#
# # Step 2: Load the model back
# loaded_model = joblib.load('trained_model_nonzero.pkl')
#
# # Step 3: Perform a test prediction
# test_prediction = loaded_model.predict(X_non_zero_no_outlier)
# mae = mean_absolute_error(y_non_zero, test_prediction)
#
# # Display the test prediction
# print("Test Prediction:", test_prediction)
# print("MAE:", mae)
#
# # Step 4: Optionally compare model parameters or attributes
# # (This step depends on the kind of model you are using)
# # Example for a RandomForestRegressor
# if hasattr(best_model, 'n_estimators'):
#     print("Original model's n_estimators:", best_model.n_estimators)
#     print("Loaded model's n_estimators:", loaded_model.n_estimators)