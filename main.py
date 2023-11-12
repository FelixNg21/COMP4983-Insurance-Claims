import pandas as pd
import numpy as np
import random
import imbalanced_dataset_regression as imb


# Fit linear reg model
# inputs:
# X: training features
# y: training outputs
# outputs:
# w: estimated weights of lin reg model
def lin_reg_fit(X, y):
    # add a column of ones to X
    X = np.c_[np.ones((X.shape[0], 1)), X]
    # compute the pseudo-inverse of X
    X_pinv = np.linalg.pinv(X)
    return X_pinv.dot(y)


# Predict using linear reg model
# inputs:
# X: test features
# w: estimated weights of lin reg model
# outputs:
# y: predicted outputs
def lin_reg_predict(X, w):
    # add a column of ones to X
    X = np.c_[np.ones((X.shape[0], 1)), X]
    return X.dot(w)


def lin_reg(data):
    # split the data into training set and test set
    train_ratio = 0.75
    # number of samples in the data_subset
    num_rows = data.shape[0]
    # shuffle the indices
    shuffled_indices = list(range(num_rows))
    random.seed(42)
    random.shuffle(shuffled_indices)

    # calculate the number of rows for training
    train_set_size = int(train_ratio * num_rows)

    # training set: take the first 'train_set_size' rows
    train_indices = shuffled_indices[:train_set_size]
    # test set: take the remaining rows
    test_indices = shuffled_indices[train_set_size:]

    # create training set and test set
    train_data = data.iloc[train_indices, :]
    test_data = data.iloc[test_indices, :]
    print(len(train_data), "training samples + ", len(test_data), "test samples")

    # prepare training features and training labels
    # features: all columns except 'ClaimAmount
    # labels: 'price' column
    train_features = train_data.drop('ClaimAmount', axis='columns', inplace=False)
    train_labels = train_data.loc[:, 'ClaimAmount']

    # prepare test features and test labels
    test_features = test_data.drop('ClaimAmount', axis='columns', inplace=False)
    test_labels = test_data.loc[:, 'ClaimAmount']

    # train a linear regression model using training data
    weights = lin_reg_fit(train_features, train_labels)
    print("Weights", weights)
    # predict new prices on test data
    price_pred = lin_reg_predict(test_features, weights)

    # compute mean absolute error
    mae = np.mean(np.abs(test_labels - price_pred))
    print('Mean Absolute Error = ', mae)

    # compute root means square error
    rmse = np.sqrt(np.mean((test_labels - price_pred) ** 2))
    print('Root Mean Squared Error = ', rmse)

    # compute coefficient of determination (aka R squared)
    total_sum_sq = np.sum((test_labels - np.mean(test_labels)) ** 2)
    res_sum_sq = np.sum((test_labels - price_pred) ** 2)
    CoD = 1 - (res_sum_sq / total_sum_sq)

    print('Coefficient of Determination = ', CoD)


# data = pd.read_csv('trainingset.csv')
# lin_reg(data)

reg = imb.Imbalanced_Dataset_Reg('trainingset.csv', 'ClaimAmount', 0.1, True)
reg.no_resampling()
reg.smoter()
reg.gauss()
reg.wercs()
# reg.plot_overall()


