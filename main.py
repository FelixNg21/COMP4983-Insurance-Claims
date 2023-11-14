import pandas as pd
import numpy as np
import random
import ridge_lasso as rl
from imbalanced_dataset_regression import ImbalancedDatasetReg as imb, Resampler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

def lin_reg_fit(X, y):
    """
    Fits a linear regression model to the given input features and target values.

    Args:
        X (numpy.ndarray): The input features of shape (n_samples, n_features).
        y (numpy.ndarray): The target values of shape (n_samples,).

    Returns:
        numpy.ndarray: The weight vector of shape (n_features + 1,).
    """
    # add a column of ones to X
    X = np.c_[np.ones((X.shape[0], 1)), X]
    # compute the pseudo-inverse of X
    X_pinv = np.linalg.pinv(X)
    return X_pinv.dot(y)

def lin_reg_predict(X, w):
    """
    Predicts the output of a linear regression model.

    Args:
        X (numpy.ndarray): The input features of shape (n_samples, n_features).
        w (numpy.ndarray): The weight vector of shape (n_features + 1,).

    Returns:
        numpy.ndarray: The predicted output of shape (n_samples,).
    """
    # add a column of ones to X
    X = np.c_[np.ones((X.shape[0], 1)), X]
    return X.dot(w)

def lin_reg(data):
    """
    Performs linear regression on the given dataset.

    Args:
        data (pandas.DataFrame): The input dataset containing the features and target variable.

    Returns:
        None
    """
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


# load data and remove first column (Row Index)
train_data = pd.read_csv('trainingset.csv').iloc[:, 1:]

test_data = pd.read_csv('testset.csv')
resampler = Resampler(train_data, 0.1)
smoter_data_label, smoter_data_feature = resampler.smoter()
gauss_data_label, gauss_data_feature = resampler.gauss()
wercs_data_label, wercs_data_feature = resampler.wercs()

smoter_data_combined = pd.concat([pd.DataFrame(smoter_data_label), pd.DataFrame(smoter_data_feature)], axis=1)
smoter_data_combined.columns = [*smoter_data_combined.columns[:-1], 'ClaimAmount']
gauss_data_combined = pd.concat([pd.DataFrame(gauss_data_label), pd.DataFrame(gauss_data_feature)], axis=1)
gauss_data_combined.columns = [*gauss_data_combined.columns[:-1], 'ClaimAmount']
wercs_data_combined = pd.concat([pd.DataFrame(wercs_data_label), pd.DataFrame(wercs_data_feature)], axis=1)
wercs_data_combined.columns = [*wercs_data_combined.columns[:-1], 'ClaimAmount']

resampled_data = [smoter_data_combined, gauss_data_combined, wercs_data_combined]
for idx, data in enumerate(resampled_data):
    rlm = rl.LinearModel(data)
    rlm.ridge(0, 20)
    rlm.lasso(0, 20)
    ridge_model = Ridge(alpha=rlm.get_ridge_alpha())
    lasso_model = Lasso(alpha=rlm.get_lasso_alpha())
    models = [ridge_model, lasso_model]
    for idx2, model in enumerate(models):
        model.fit(data.iloc[:, :-1], data.iloc[:, -1])
        predict = model.predict(test_data.iloc[:,1:])
        pd.DataFrame(predict).to_csv(f'submission_resamp{idx}_model{idx2}.csv', index=False)

# lin_reg(data)

# # determine alpha values for ridge and lasso regression
# rl = rl.LinearModel(data)
# rl.ridge(0, 10)
# rl.lasso(0, 10)
# ridge_alpha = rl.get_ridge_alpha()
# lasso_alpha = rl.get_lasso_alpha()

# regression_model_default = RandomForestRegressor(n_estimators=50, max_features=0.5, n_jobs=-1,
#                                                  random_state=0)
# regression_model_ridge = Ridge(alpha=ridge_alpha)
# regression_model_lasso = Lasso(alpha=lasso_alpha)
#
# models = [regression_model_default, regression_model_ridge, regression_model_lasso]

# determine best values for smoter/gauss/wercs resampling techniques
# for model in models:
#     reg = imb.ImbalancedDatasetReg('trainingset.csv', 'ClaimAmount', 0.1, model)
#     reg.no_resampling()
#     reg.smoter()
#     reg.gauss()
#     reg.wercs()


# with resampling

