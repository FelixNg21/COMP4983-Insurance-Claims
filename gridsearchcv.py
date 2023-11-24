from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import joblib

num_to_remove = 30000

# Load the dataset
data = pd.read_csv('trainingset.csv')
data = data[data['ClaimAmount'] < 4646.23]
data = data.drop(data[data['ClaimAmount'] == 0].sample(frac=0.8).index)
print(data)
print(data.shape)
x_data = data.iloc[:, 1:-1]  # Features
y_data = data['ClaimAmount']  # Labels

# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

def do_gridsearch():
    # Create a RandomForestRegressor
    rf = RandomForestRegressor()

    # Define the hyperparameter grid to search over
    param_grid = {
        'n_estimators': [50, 100, 300],
        'max_depth': [None, 20, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the model to the data
    grid_search.fit(x_train, y_train)

    # Print the best hyperparameters
    print("Best hyperparameters:", grid_search.best_params_)
    # build model from best_params
    best_params = {'max_depth': 40, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 300}


    # Get the best model
    # best_model = grid_search.best_estimator_
    best_model = RandomForestRegressor(random_state=42, **best_params)


    # You can also access other information such as the best score
    print("Best negative mean squared error:", grid_search.best_score_)
    return best_params, best_model
# best_params, best_model = do_gridsearch()
# best_params = {'max_depth': 40, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 300}

best_params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
best_model = RandomForestRegressor(random_state=42, **best_params)
best_model.fit(x_train, y_train)

grid_predictions = best_model.predict(x_test)

# get csv of test and predicted values
y_test.to_csv('gridsearch.csv')
gridsearch = pd.read_csv('gridsearch.csv')
gridsearch['predicted'] = grid_predictions
gridsearch.to_csv('gridsearch.csv')

print("mae: ", mean_absolute_error(y_test, grid_predictions))
print("mse: ", mean_squared_error(y_test, grid_predictions))
# print("rmse: ", np.sqrt(np.mean((y_test - grid_predictions) ** 2)))
print("r2: ", r2_score(y_test, grid_predictions))
best_model.fit(x_data, y_data)
joblib.dump(best_model, 'felix_randomforestregressor.pkl')




