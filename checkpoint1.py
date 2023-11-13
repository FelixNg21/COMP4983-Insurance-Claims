import random
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
# from pyhurdle import Hurdle

data = pd.read_csv('trainingset.csv')

def preprocess(data):
    data_subset = data
    # print(data.groupby('ClaimAmount').count())

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
    train_features = train_data.drop('ClaimAmount', axis='columns', inplace=False)
    train_labels = train_data['ClaimAmount']

    # prepare test features and test labels
    test_features = test_data.drop('ClaimAmount', axis='columns', inplace=False)
    test_labels = test_data['ClaimAmount']
    
    return train_features, train_labels, test_features, test_labels
    

def lin_reg_model(train_features, train_labels, test_features, test_labels):

    lin_reg = LinearRegression()

    # train a linear regression model using training data
    weights = lin_reg.fit(train_features, train_labels)

    # predict new prices on test data
    price_pred = lin_reg.predict(test_features)

    # compute mean absolute error
    mae = np.mean(abs(test_labels - price_pred))
    print('Mean Absolute Error = ', mae)

    # compute root means square error
    rmse = np.sqrt(np.mean((test_labels - price_pred) ** 2))
    print('Root Mean Squared Error = ', rmse)

    # compute coefficient of determination (aka R squared)
    total_sum_sq = np.sum((test_labels - np.mean(test_labels)) ** 2)
    res_sum_sq = np.sum((test_labels - price_pred) ** 2)
    CoD = 1 - (res_sum_sq / total_sum_sq)

    print('Coefficient of Determination = ', CoD)
    
    return test_labels, price_pred, mae

def zero_inflated_poisson_reg_model(train_features, train_labels, test_features, test_labels):
    X = sm.add_constant(train_features)
    zip_model = sm.ZeroInflatedPoisson(train_labels, X).fit()
    # print(zip_reg.summary())
    data_to_predict = sm.add_constant(test_features)
    price_pred = zip_model.predict(data_to_predict)
    mae = mean_absolute_error(test_labels, price_pred)
    print("\n\n\nPredicted prices: ")
    print(price_pred)
    print("mae using zero inflated poisson regression", mae)
    return test_labels, price_pred, mae

def zero_inflated_negative_binomial_reg_model(train_features, train_labels, test_features, test_labels):
    X = sm.add_constant(train_features)
    zinb_model = sm.ZeroInflatedNegativeBinomialP(train_labels, X).fit()
    # print(zip_reg.summary())
    data_to_predict = sm.add_constant(test_features)
    price_pred = zinb_model.predict(data_to_predict)
    mae = mean_absolute_error(test_labels, price_pred)
    print("\n\n\nPredicted prices: ")
    print(price_pred.values)
    print(mae)
    return test_labels, price_pred, mae

# def two_part(train_features, train_labels, test_features, test_labels):
#     hurdle_model = Hurdle()
#     hurdle_model.fit(train_features, train_labels)
#     price_pred = hurdle_model.predict(test_features)
#     mae = mean_absolute_error(test_labels, price_pred)
#     print("\n\n\nPredicted prices: ")
#     print(price_pred)
#     print(mae)
#     return test_labels, price_pred, mae

def generate_csv_test_labels(csv_file_name, test_labels):
    test_labels.to_csv(csv_file_name, index=False, header=False)
    
def add_to_csv(csv_file_name, price_pred):
    df = pd.read_csv(csv_file_name, header=None)
    df['ClaimAmount'] = price_pred
    df.to_csv(csv_file_name, index=False, header=False)

### preprocess data ###
train_features, train_labels, test_features, test_labels = preprocess(data)

### linear regression ####
# test_labels, price_pred, mae = lin_reg_model(train_features, train_labels, test_features, test_labels)
# generate_csv_test_labels('lin_reg.csv', test_labels)
# add_to_csv('lin_reg.csv', price_pred)

### zero inflated poisson regression ###
test_labels, price_pred, mae = zero_inflated_poisson_reg_model(train_features, train_labels, test_features, test_labels)
generate_csv_test_labels('zip_reg.csv', test_labels)
add_to_csv('zip_reg.csv', price_pred.values)

### zero infalted negative binomial ###
# test_labels, price_pred, mae = zero_inflated_poisson_reg_model(train_features, train_labels, test_features, test_labels)
# generate_csv_test_labels('zinb_reg.csv', test_labels)
# add_to_csv('zinb_reg.csv', price_pred.values)

# def two_part():
#     subset_data = data
#     subset_data['non_zero_claim'] = (subset_data['ClaimAmount'] > 0).astype(int)
#     train_features, train_labels, test_features, test_labels = preprocess(subset_data)
    
#     non_zero_train = subset_data['non_zero_claim'].iloc[train_features.index]
#     non_zero_test = subset_data['non_zero_claim'].iloc[test_features.index]
    
#     # Binary Logistic Regression
#     logistic_model = LogisticRegression(max_iter=1000)
#     logistic_model.fit(train_features, non_zero_train)
    
#     # predict claim is 0 or not
#     predictions_binary = logistic_model.predict(test_features)
    
#     # linear regression on non-zero claims
#     non_zero_indices = non_zero_test[non_zero_test == 1].index
#     linear_model = LinearRegression()
#     linear_model.fit(train_features.loc[non_zero_indices], train_labels.loc[non_zero_indices])
    
#     predictions_continuous = linear_model.predict(test_features.loc[non_zero_indices])
    
#     predictions_combined = np.zeros_like(test_labels, dtype=float)
#     predictions_combined[non_zero_indices] = predictions_continuous
    
#     mae = mean_absolute_error(test_labels, predictions_combined)
    
#     print(mae)
    
# two_part()
    