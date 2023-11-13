
import pandas as pd
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('trainingset.csv')

def preprocess(data, label):
    data_subset = data
    data_subset = data_subset.drop('rowIndex', axis='columns', inplace=False)
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

# logistic regression
def logistic_regression(data):
  train_features, train_labels, test_features, test_labels = insert_binary_column(data)  
  
  logistic_reg_model = LogisticRegression(max_iter=1000)
  logistic_reg_model.fit(train_features, train_labels)
  log_reg_predictions = logistic_reg_model.predict(test_features)
  
  print(log_reg_predictions)
  
  
  
  # label = 'ClaimAmount'
  # train_features, train_labels, test_features, test_labels = preprocess(data, label)
  

logistic_regression(data)
  