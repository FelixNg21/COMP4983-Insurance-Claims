
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('trainingset.csv')

scaler = StandardScaler()

# remove all outliers from dataset
dataset = dataset[dataset['ClaimAmount'] < 4000]

# remove all claimamounts that are 0 from dataset
dataset_no0 = dataset[dataset['ClaimAmount'] > 0]

# leave only 10% of the 0 claimamounts
dataset_5_percent_zero = pd.concat([dataset[dataset['ClaimAmount'] == 0].sample(frac=0.0025, random_state=42),dataset_no0])

# X_no0 = dataset_no0.iloc[:, 1:-1]
# y_no0 = dataset_no0['ClaimAmount']
# X_no0 = scaler.fit_transform(X_no0)
# X_train, X_test, y_train, y_test = train_test_split(X_no0, y_no0, test_size=0.2, random_state=42)


X_5_percent_zero = dataset_5_percent_zero.iloc[:, 1:-1]
y_5_percent_zero = dataset_5_percent_zero['ClaimAmount']
print(dataset_5_percent_zero['ClaimAmount'].value_counts())
print(dataset_5_percent_zero['ClaimAmount'].shape)
X_5_percent_zero = scaler.fit_transform(X_5_percent_zero)
X_train, X_test, y_train, y_test = train_test_split(X_5_percent_zero, y_5_percent_zero, test_size=0.2, random_state=42)


# RF = RandomForestRegressor()
# param_grid = { # for random forest
#     'n_estimators': [50, 100, 200],  # Number of trees in the forest
#     'max_features': ['sqrt'],  # Number of features to consider at every split # , 'log2', None, 0.2, 0.4, 0.6, 0.8, 8,9,10,11,12,13,14,15,16,17,18
#     'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
#     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
# }
# grid_RF = GridSearchCV(estimator=RF, param_grid=param_grid, cv=4, n_jobs=-1, scoring='neg_mean_absolute_error')
# grid_RF.fit(X_train, y_train)
# # Print the best hyperparameters
# print("Best Hyperparameters:", grid_RF.best_params_)
# # Use the best model to make predictions on the test set
# best_model = grid_RF.best_estimator_
# print("Best Modal:", best_model)
# y_pred = best_model.predict(X_test)
# pd.DataFrame(y_pred).to_csv('sarah_y_pred.csv')
# pd.DataFrame(y_test).to_csv('sarah_y_test.csv')
#
# # Evaluate the performance of the best model
# best_score = grid_RF.best_score_
# print("Best Score:", best_score)
#
# best_model.fit(X_5_percent_zero, y_5_percent_zero)
# joblib.dump(best_model, 'sarah_best_model_rf_gridsearch_mae_tuned_FORFUN.joblib')

GBR = HistGradientBoostingRegressor()

# param_grid = { # for gradient boosting
#     'n_estimators': [50, 100, 150],  # Number of boosting stages to be run
#     'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
#     'max_depth': [3, 4, 5, None],  # Maximum depth of the individual trees
#     # 'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
#     'subsample': [0.8, 1.0],  # Fraction of samples used for fitting the trees
#     # 'max_features': ['sqrt', 'log2', None]  # Number of features to consider for the best split
# }

param_grid = { # for histogram gradient boosting
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
    'max_iter': [50, 100, 150],  # Number of boosting iterations
    'max_depth': [3, 4, 5],  # Maximum depth of the individual trees
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_bins': [100, 150, 200, 255],  # Maximum number of bins for histogram splitting
    'l2_regularization': [0.0, 0.1, 0.2],  # L2 regularization term on weights
    'max_leaf_nodes': [None, 31, 63],  # Maximum number of leaves for each tree
    'random_state': [42]  # Set to a specific value for reproducibility
}


grid_GBR = GridSearchCV(estimator=GBR, param_grid=param_grid, cv=4, n_jobs=-1, scoring='neg_mean_absolute_error')
grid_GBR.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_GBR.best_params_)

# Use the best model to make predictions on the test set
best_model = grid_GBR.best_estimator_
print("Best Mode:", best_model)
y_pred = best_model.predict(X_test)
pd.DataFrame(y_pred).to_csv('sarah_y_pred.csv')
pd.DataFrame(y_test).to_csv('sarah_y_test.csv')
# Evaluate the performance of the best model
best_score = grid_GBR.best_score_
print("Best Score:", best_score)

best_model.fit(X_5_percent_zero, y_5_percent_zero)
joblib.dump(best_model, 'sarah_best_model_hgbs_gridsearch_mae_tuned.joblib')



