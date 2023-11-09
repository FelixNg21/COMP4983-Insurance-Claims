import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import resreg
import itertools
from sklearn.datasets import fetch_california_housing as dataset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')


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


data = pd.read_csv('trainingset.csv')
# lin_reg(data)


# SMOTER


# define what is a low and high claim
bins = [0.1]

CACHE = {}


def implementML(X_train, y_train, X_test, y_test, reg, over=None, k=None):
    reg.fit(X_train, y_train)  # fit regressor
    y_pred = reg.predict(X_test)

    if over is not None and k is not None:
        df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        df.to_csv(f'{over}_{k}_output.csv')
    r2 = r2_score(y_test, y_pred)
    rel_test = resreg.sigmoid_relevance(y_test, cl=None, ch=bins[0])  # relevance values of y_test
    rel_pred = resreg.sigmoid_relevance(y_pred, cl=None, ch=bins[0])  # relevance values of y_pred
    f1 = resreg.f1_score(y_test, y_pred, error_threshold=0.5, relevance_true=rel_test,
                         relevance_pred=rel_pred, relevance_threshold=0.5)
    msebin = resreg.bin_performance(y_test, y_pred, bins=bins, metric='mse')
    return r2, f1, msebin


ticks_font = {'size': '12'}
label_font = {'size': '14'}
title_font = {'size': '16'}


def plotPerformance(msebin, msebinerr, f1, r2, title):
    plt.bar(range(2), msebin, yerr=msebinerr, width=0.4, capsize=3, color='royalblue',
            linewidth=1, edgecolor='black')
    plt.xlim(-0.5, len(bins) + 0.5)
    plt.xticks(range(2), ['< {0}'.format(bins[0]), '≥ {0}'.format(bins[0])], **ticks_font)
    plt.yticks(**ticks_font)
    plt.ylabel('Mean Squared Error (MSE)', **label_font)
    plt.xlabel('Target value range', **label_font)
    title = title + '\nf1={0}, r2={1}'.format(round(f1, 3), round(r2, 3))
    plt.title(title, **title_font)
    plt.show()
    plt.close()


def no_resampling():
    X = data.drop('ClaimAmount', axis='columns', inplace=False)
    y = data.loc[:, 'ClaimAmount']
    np.random.seed(seed=0)
    sample = np.random.choice(range(len(y)), 500)
    X, y = X.loc[sample, :], y[sample]
    # Empty list for storing results
    r2s, f1s, msebins = [], [], []

    # Fivefold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    for train_index, test_index in kfold.split(X):
        X_train, y_train = X.iloc[train_index, :], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index, :], y.iloc[test_index]
        reg = RandomForestRegressor(n_estimators=50, max_features=0.5, n_jobs=-1, random_state=0)
        r2, f1, msebin = implementML(X_train, y_train, X_test, y_test,
                                     reg)  # Fit regressor and evaluate performance
        r2s.append(r2);
        f1s.append(f1);
        msebins.append(msebin)

    # Average performance
    r2, f1, msebin = np.mean(r2s), np.mean(f1s), np.mean(msebins, axis=0)
    # Standard error of the mean
    r2err, f1err, msebinerr = np.std(r2s) / np.sqrt(5), np.std(f1s) / np.sqrt(5), \
                              np.std(msebins, axis=0) / np.sqrt(5)
    # View performance
    plotPerformance(msebin, msebinerr, f1, r2, title='No resampling (None)')

    # Save performance results
    CACHE['None'] = [r2, f1, msebin, r2err, f1err, msebinerr]


def smoter():
    X = data.drop('ClaimAmount', axis='columns', inplace=False)
    y = data.loc[:, 'ClaimAmount']
    np.random.seed(seed=0)
    sample = np.random.choice(range(len(y)), 500)
    X, y = X.loc[sample, :], y[sample]

    # Parameters
    overs = ['balance', 'average', 'extreme']
    ks = [5, 10, 15]  # nearest neighbors
    params = list(itertools.product(overs, ks))

    # Empty lists for storing results
    r2store, f1store, msebinstore = [], [], []
    r2errstore, f1errstore, msebinerrstore = [], [], []

    # Grid search
    for over, k in params:
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        r2s, f1s, msebins = [], [], []

        # Fivefold cross validation
        for train_index, test_index in kfold.split(X):
            X_train, y_train = X.iloc[train_index, :], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index, :], y.iloc[test_index]

            # Resample training data (SMOTER)
            relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=bins[0])
            X_train, y_train = resreg.smoter(X_train, y_train, relevance,
                                             relevance_threshold=0.5, k=k, over=over,
                                             random_state=0)

            # Fit regressor and evaluate performance
            reg = RandomForestRegressor(n_estimators=50, max_features=0.5, n_jobs=-1,
                                        random_state=0)
            r2, f1, msebin = implementML(X_train, y_train, X_test, y_test, reg, over, k)
            r2s.append(r2)
            f1s.append(f1)
            msebins.append(msebin)
        r2, f1, msebin = np.mean(r2s), np.mean(f1), np.mean(msebins, axis=0)
        r2err, f1err, msebinerr = np.std(r2s) / np.sqrt(5), np.std(f1s) / np.sqrt(5), \
                                  np.std(msebins, axis=0) / np.sqrt(5)

        # Store grid search results
        r2store.append(r2)
        f1store.append(f1)
        msebinstore.append(msebin)
        r2errstore.append(r2err)
        f1errstore.append(f1err)
        msebinerrstore.append(msebinerr)

    # Determine the best parameters
    best = np.argsort(f1store)[-1]  # Which is the best
    print('''Best parameters:
        over={0}; k={1}'''.format(params[best][0], params[best][1]))
    f1, r2, msebin = f1store[best], r2store[best], msebinstore[best]
    f1err, rerr, msebinerr = f1errstore[best], r2errstore[best], msebinerrstore[best]

    # Save results
    CACHE['SMOTER'] = [r2, f1, msebin, r2err, f1err, msebinerr]

    # Plot results
    plotPerformance(msebin, msebinerr, f1, r2, title='SMOTER')


no_resampling()
smoter()
print(CACHE)
