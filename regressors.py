
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, SGDRegressor
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import resreg


data = pd.read_csv('trainingset.csv')

def preprocess(data, label):
    data_subset = data
    data_subset = data_subset.drop('rowIndex', axis='columns', inplace=False)

    train_ratio = 0.75

    # number of samples in the data_subset
    num_rows = data_subset.shape[0]

    # shuffle the data
    shuffled_indices = list(range(num_rows))

    train_set_size = int(num_rows * train_ratio)
    train_indices = shuffled_indices[:train_set_size]

    # test set: take the remaining rows
    test_indices = shuffled_indices[train_set_size:]

    # create training set and test set
    train_data = data.iloc[train_indices, :]
    test_data = data.iloc[test_indices, :]
    print(len(train_data), "training samples + ", len(test_data), "test samples")

    # prepare training features and training labels
    train_features = train_data.drop(label, axis='columns', inplace=False)
    train_labels = train_data[label]

    # prepare test features and test labels
    test_features = test_data.drop(label, axis='columns', inplace=False)
    test_labels = test_data[label]
    
    return train_features, train_labels, test_features, test_labels
  

def insert_binary_column(data):
  data_subset = data
  data_subset['non_zero_claim'] = (data_subset['ClaimAmount'] > 0).astype(int)
  return preprocess(data_subset, 'non_zero_claim')


# class nn_regression(Sequential()):
  
#   def __init__(self):
#     super(self).__init__()
    
#     # input_dim is the number of features
#     self.add(Dense(64, input_dim=18, activation='relu'))
#     self.add(Dropout(0.2))
#     self.add(Dense(32, activation='relu'))
#     self.add(Dropout(0.2))
#     self.add(Dense(1, activation='linear')) # output layer for regression
#     self.compile(optimizer='adam', metrics=['mse', 'mae', 'accuracy'], loss='binary_crossentropy')

# standardize the features
def standardize_features(train_features, test_features):
  scaler = StandardScaler()
  train_features_scaled = scaler.fit_transform(train_features)
  test_features_scaled = scaler.transform(test_features)
  
  return train_features_scaled, test_features_scaled
  
# MLP Classifier - has an underlying neural network
# Accuracy is 0.8694 without standardizing features, 
# Accuracy is 0.9229 with standardizing features
def mlp_classifier(data):
  print('\nMLP classifier')
  train_features, train_labels, test_features, test_labels = insert_binary_column(data)
  train_features = train_features.drop('ClaimAmount', axis='columns', inplace=False)
  test_features = test_features.drop('ClaimAmount', axis='columns', inplace=False)
  
  # hidden_layer_sizes = (100, 100, 100) means 3 hidden layers with 100 neurons each
  num_epochs = 5000 # max_iter is the number of epochs
  classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=num_epochs)
  
  # balance the dataset using smote
  smote = SMOTE(sampling_strategy='auto')
  train_features, train_labels = smote.fit_resample(train_features, train_labels)
  
  # comment out the following line to not standardize the features
  train_features, test_features = standardize_features(train_features, test_features)
  
  classifier.fit(train_features, train_labels)
  test_predictions = classifier.predict(test_features)
  
  print("\nTest labels:")
  print(Counter(test_labels))
  print("\nPredicted labels:")
  print(Counter(test_predictions))
  print("Accuracy of MLPClassifier : ", classifier.score(test_features, test_labels))



# kfold validation for regression models
def k_fold_regress(data, regression_models):
  train_features, train_labels, test_features, test_labels = preprocess(data, 'ClaimAmount')
  
  train_features, test_features = standardize_features(train_features, test_features)
  
  k_folds = 5
  kf = KFold(n_splits=k_folds, shuffle=True)
  lambdas = [10 ** i for i in range(-3, 11)]
  
  for model in regression_models:
    print("\nModel:", type(model).__name__)
    
    model.fit(train_features, train_labels)
    
    model_name = type(model).__name__
    joblib.dump(model, f'{model_name}_model.joblib')
    
    if isinstance(model, (Ridge, Lasso, ElasticNet)):
      for alpha in lambdas:
        model.set_params(alpha=alpha)
        
        scores = cross_val_score(model, train_features, train_labels, cv=kf, scoring='neg_mean_absolute_error')
        scores = -scores
        print("MAE for each fold:", scores)
        print("MAE:", scores.mean())
    
    else:
      scores = cross_val_score(model, train_features, train_labels, cv=kf, scoring='neg_mean_absolute_error')
      scores = -scores
      print("MAE for each fold:", scores)
      print("MAE:", scores.mean())
      
      
      # model.fit(train_features, train_labels)
      # predictions = model.predict(test_features)
      
      # print("Accuracy:", model.score(test_features, test_labels))
      # print("MAE:", mean_absolute_error(test_labels, predictions))
      
      

def regress(data, regression_models):
  train_features, train_labels, test_features, test_labels = preprocess(data, 'ClaimAmount')
  
  train_features, test_features = standardize_features(train_features, test_features)
  
  lambdas = [10 ** i for i in range(-3, 11)]
  
  for model in regression_models:
    print("\nModel:", type(model).__name__)
    
    model.fit(train_features, train_labels)
    
    model_name = type(model).__name__
    joblib.dump(model, f'{model_name}_model.joblib')
    
    if isinstance(model, (Ridge, Lasso, ElasticNet)):
      for alpha in lambdas:
        model.set_params(alpha=alpha)
        
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        
        print("Alpha:", alpha, "MAE:", mean_absolute_error(test_labels, predictions))
        # print("Accuracy:", model.score(test_features, test_labels))
    
    else:
      model.fit(train_features, train_labels)
      predictions = model.predict(test_features)
      
      print("MAE", mean_absolute_error(test_labels, predictions))
      # print("Accuracy:", model.score(test_features, test_labels))
  
  
def bagging_regress_voting(data, regression_models):
  train_features, train_labels, test_features, test_labels = preprocess(data, 'ClaimAmount')
  
  train_features, test_features = standardize_features(train_features, test_features)
  
  models = [(type(model).__name__, model) for model in regression_models]
  
  ensemble_model = VotingRegressor(estimators=models)
  ensemble_model.fit(train_features, train_labels)
  predictions = ensemble_model.predict(test_features)
  
  # k fold cross validation
  k_folds = 5
  kf = KFold(n_splits=k_folds, shuffle=True)

  scores = cross_val_score(ensemble_model, train_features, train_labels, cv=kf, scoring='neg_mean_absolute_error')
  scores = -scores
  print("MAE for each fold:", scores)
  print("MAE:", scores.mean())
  
  # no x validation
  print("MAE:", mean_absolute_error(test_labels, predictions))
  
  
# preprocess data
# train_features, train_labels, test_features, test_labels = preprocess(data, 'ClaimAmount')

# classifer 
# mlp_classifier(data)

# k fold cross validation, imbalanced dataset
regression_models = [
    LinearRegression(), # 202
    Ridge(alpha=1.0), # 203
    Lasso(alpha=1.0), # 203
    ElasticNet(alpha=1.0, l1_ratio=0.5), # 203
    DecisionTreeRegressor(), # 188
    RandomForestRegressor(n_estimators=100), # 193
    GradientBoostingRegressor(n_estimators=100), # 206
    SVR(kernel='linear'), # 106
    SGDRegressor(), # 196
    KNeighborsRegressor(), # 187
    # nn_regression().model
  ]

# just regression
regression_models = [
    LinearRegression(), # 185
    Ridge(alpha=1.0), # 185
    Lasso(alpha=1.0), # 185
    ElasticNet(alpha=1.0, l1_ratio=0.5), # 187
    DecisionTreeRegressor(), # 884
    RandomForestRegressor(n_estimators=100), # 566 # bagging - an ensemble of decision trees
    GradientBoostingRegressor(n_estimators=100), #357 # boosting - gradient boosting
    SVR(kernel='linear'), # 92
    SGDRegressor(), # 264
    KNeighborsRegressor(), # 181
    # nn_regression().model
  ]


# k_fold_regress(data, regression_models)
# regress(data, regression_models)

models = [RandomForestRegressor(n_estimators=100), GradientBoostingRegressor(n_estimators=100), LinearRegression()]
bagging_regress_voting(data, models)




