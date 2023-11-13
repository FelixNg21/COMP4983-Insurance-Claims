import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model_with_cross_validation(model, X, y, n_splits=5):
    mse_scores = cross_val_score(model, X, y, cv=n_splits, scoring='neg_mean_squared_error')
    mae_scores = cross_val_score(model, X, y, cv=n_splits, scoring='neg_mean_absolute_error')

    mean_mse = np.mean(mse_scores)
    mean_mae = np.mean(mae_scores)

    print(f'{model.__class__.__name__}: Mean Squared Error: {-mean_mse}, Mean Absolute Error: {-mean_mae}')

# Load and prepare the dataset
data = pd.read_csv('trainingset.csv')
data = data.drop(data.columns[0], axis=1)
non_zero_data = data[data['ClaimAmount'] > 0]
non_zero_data = non_zero_data[non_zero_data['ClaimAmount'] < 5000]
X_non_zero = non_zero_data.iloc[:, 1:-1]
y_non_zero = non_zero_data['ClaimAmount']

# quit()
# Standardize features
scaler = StandardScaler()
X_non_zero_scaled = scaler.fit_transform(X_non_zero)

# Define models to evaluate
models = [
    Lasso(alpha=0.001),
    Ridge(alpha=0.001),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR(),
    HuberRegressor()
]

# Evaluate each model using cross-validation
for model in models:
    evaluate_model_with_cross_validation(model, X_non_zero_scaled, y_non_zero)
