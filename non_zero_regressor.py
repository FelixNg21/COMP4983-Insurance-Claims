import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
import joblib  # For saving the model

def evaluate_model_with_cross_validation(model, X, y, n_splits=5):
    mse_scores = cross_val_score(model, X, y, cv=n_splits, scoring='neg_mean_squared_error')
    mae_scores = cross_val_score(model, X, y, cv=n_splits, scoring='neg_mean_absolute_error')

    mean_mse = np.mean(mse_scores)
    mean_mae = np.mean(mae_scores)

    return -mean_mse, -mean_mae

# Load and prepare the dataset
data = pd.read_csv('trainingset.csv')
non_zero_data = data[data['ClaimAmount'] > 0]

#######
# Check if we need to remove outliers!!!
non_zero_data = non_zero_data[non_zero_data['ClaimAmount'] < 5000]
X_non_zero = non_zero_data.iloc[:, 1:-1]
y_non_zero = non_zero_data['ClaimAmount']

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

# Evaluate each model and track their performance
model_performance = {}
for model in models:
    mse, mae = evaluate_model_with_cross_validation(model, X_non_zero_scaled, y_non_zero)
    model_performance[model.__class__.__name__] = (mse, mae)

# Identify the best model (lowest MAE)
best_model_name = min(model_performance, key=lambda k: model_performance[k][1])
print(best_model_name)
best_model = [model for model in models if model.__class__.__name__ == best_model_name][0]

# Fit the best model on the entire dataset
best_model.fit(X_non_zero_scaled, y_non_zero)


# Step 1: Save the model
joblib.dump(best_model, 'trained_model_nonzero.pkl')

# Step 2: Load the model back
loaded_model = joblib.load('trained_model_nonzero.pkl')

# Step 3: Perform a test prediction
test_prediction = loaded_model.predict(X_non_zero)
mae = mean_absolute_error(y_non_zero, test_prediction)

# Display the test prediction
print("Test Prediction:", test_prediction)
print("MAE:", mae)

# Step 4: Optionally compare model parameters or attributes
# (This step depends on the kind of model you are using)
# Example for a RandomForestRegressor
if hasattr(best_model, 'n_estimators'):
    print("Original model's n_estimators:", best_model.n_estimators)
    print("Loaded model's n_estimators:", loaded_model.n_estimators)