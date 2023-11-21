from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import pandas as pd
from sklearn.metrics import classification_report

num_to_remove = 30000

# Load the dataset
data = pd.read_csv('trainingset.csv')
data = data.drop(data[data['ClaimAmount'] == 0].sample(frac=0.8).index)
print(data)
print(data.shape)
x_data = data.iloc[:, 1:-1]  # Features
y_data = data['ClaimAmount']  # Labels

# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


# Create a RandomForestRegressor
rf = RandomForestRegressor()

# Define the hyperparameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50, 60],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the model to the data
grid_search.fit(x_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# You can also access other information such as the best score
print("Best negative mean squared error:", grid_search.best_score_)

best_model.fit(x_data, y_data)

grid_predictions = best_model.predict(x_test)

print(classification_report(y_test, grid_predictions))




