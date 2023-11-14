import numpy as np
import itertools
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

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
  
train_features, train_labels, test_features, test_labels = preprocess(data, 'ClaimAmount')




