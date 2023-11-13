
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

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
    # print(len(train_data), "training samples + ", len(test_data), "test samples")

    # prepare training features and training labels
    train_features = train_data.drop(label, axis='columns', inplace=False)
    train_features = train_features.drop('rowIndex', axis='columns', inplace=False)
    train_labels = train_data[label]

    # prepare test features and test labels
    test_features = test_data.drop(label, axis='columns', inplace=False)
    test_features = test_features.drop('rowIndex', axis='columns', inplace=False)
    test_labels = test_data[label]
    test_labels.to_csv('non_zero_claim.csv', index=False, header=False)
    
    return train_features, train_labels, test_features, test_labels
  
def insert_binary_column(data):
  data_subset = data
  data_subset['non_zero_claim'] = (data_subset['ClaimAmount'] > 0).astype(int)
  return preprocess(data_subset, 'non_zero_claim')

# logistic regression (1 layer NN)- PREDICTS ALL 0'S FOR SOME REASON!!!
# increased the max_iter to 5000 to avoid convergence warning
def logistic_reg(data):
  print('\nLogistic regression')
  train_features, train_labels, test_features, test_labels = insert_binary_column(data)  
  train_features = train_features.drop('ClaimAmount', axis='columns', inplace=False)
  test_features = test_features.drop('ClaimAmount', axis='columns', inplace=False)
  
  # balance the dataset using smote
  smote = SMOTE(sampling_strategy='auto')
  train_features, train_labels = smote.fit_resample(train_features, train_labels)
  
  logistic_reg_model = LogisticRegression(max_iter=5000)
  logistic_reg_model.fit(train_features, train_labels)
  log_reg_predictions = logistic_reg_model.predict(test_features)
  
  df = pd.read_csv('non_zero_claim.csv', header=None)
  df['predictions'] = log_reg_predictions
  df.to_csv('non_zero_claim.csv', index=False, header=False)
  test_labels_zeros = (test_labels == 0).sum()
  test_labels_ones = (test_labels == 1).sum()
  print("Test labels:")
  print("# 0's:", test_labels_zeros)
  print("# 1's:", test_labels_ones)
  
  pred_zeros = (log_reg_predictions == 0).sum()
  pred_ones = (log_reg_predictions == 1).sum()
  print("\nPredicted labels:")
  print("# 0's:", pred_zeros)
  print("# 1's:", pred_ones)
  
  mean_accuracy = logistic_reg_model.score(test_features, test_labels)
  print('\nMean accuracy:', mean_accuracy)
  
  return log_reg_predictions

def mlp_classifier(data):
  print('\nNeural network')
  train_features, train_labels, test_features, test_labels = insert_binary_column(data)
  train_features = train_features.drop('ClaimAmount', axis='columns', inplace=False)
  test_features = test_features.drop('ClaimAmount', axis='columns', inplace=False)
  
  classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=5000)
  
  # balance the dataset using smote
  smote = SMOTE(sampling_strategy='auto')
  train_features, train_labels = smote.fit_resample(train_features, train_labels)
  
  classifier.fit(train_features, train_labels)
  test_predictions = classifier.predict(test_features)
  # cm = confusion_matrix(test_labels, test_predictions)
  print("Accuracy of MLPClassifier : ", classifier.score(test_features, test_labels))


# random forest classifier
# 96% precision on 0's, 47% precision on 1's, macro avg 72% precision, weighted avg 94% precision
def random_forest_classifier(data):
  print('\nRandom forest classifier')
  train_features, train_labels, test_features, test_labels = insert_binary_column(data)  
  train_features = train_features.drop('ClaimAmount', axis='columns', inplace=False)
  test_features = test_features.drop('ClaimAmount', axis='columns', inplace=False)
  
  weights = {0: 16672, 1: 828}
  rf_classifier = RandomForestClassifier(class_weight=weights)
  rf_classifier.fit(train_features, train_labels)
  test_predictions = rf_classifier.predict(test_features)
  print(classification_report(test_labels, test_predictions))
  
  return test_predictions
  
  
# uses random undersampling and random oversampling
# 98% precision 0's, 11% precision on 1's, macro avg 55% precision, weighted avg 94% precision
def imbalanced_learn(data):
  print('\nImbalanced learn using random undersampling and random oversampling')
  train_features, train_labels, test_features, test_labels = insert_binary_column(data)  
  train_features = train_features.drop('ClaimAmount', axis='columns', inplace=False)
  test_features = test_features.drop('ClaimAmount', axis='columns', inplace=False)  
  print('\nBefore resampling')
  print("# 0's", (train_labels == 0).sum())
  print("# 1's", (train_labels == 1).sum())
  
  rus = RandomUnderSampler(sampling_strategy='auto')
  train_features_resampled, train_labels_resampled = rus.fit_resample(train_features, train_labels)
  print('\nAfter random under sampling resampled')
  print("# 0's", (train_labels_resampled == 0).sum())
  print("# 1's", (train_labels_resampled == 1).sum())
  
  ros = RandomOverSampler(sampling_strategy='auto')
  train_features_resampled, train_labels_resampled = ros.fit_resample(train_features_resampled, train_labels_resampled)
  print('\nAfter random over sampling resampled')
  print("# 0's", (train_labels_resampled == 0).sum())
  print("# 1's", (train_labels_resampled == 1).sum())
  
  rf_classifier = RandomForestClassifier()
  rf_classifier.fit(train_features_resampled, train_labels_resampled)
  test_predictions = rf_classifier.predict(test_features)
  print(classification_report(test_labels, test_predictions))
  
  return test_predictions

# using smote to balance dataset
# 97% precision on 0's, 38% precision on 1's, macro avg 67% precision, weighted avg 94% precision
def imbalanced_dataset_smote(data):
  print('\nImbalanced dataset using smote')
  train_features, train_labels, test_features, test_labels = insert_binary_column(data)  
  train_features = train_features.drop('ClaimAmount', axis='columns', inplace=False)
  test_features = test_features.drop('ClaimAmount', axis='columns', inplace=False)
  print('\nBefore resampling')
  print("# 0's", (train_labels == 0).sum())
  print("# 1's", (train_labels == 1).sum())
  
  smote = SMOTE(sampling_strategy='auto')
  train_features_resampled, train_labels_resampled = smote.fit_resample(train_features, train_labels)
  print('\nAfter smote resampled')
  print("# 0's", (train_labels_resampled == 0).sum())
  print("# 1's", (train_labels_resampled == 1).sum())
  
  rf_classifier = RandomForestClassifier()
  rf_classifier.fit(train_features_resampled, train_labels_resampled)
  test_predictions = rf_classifier.predict(test_features)
  
  print(classification_report(test_labels, test_predictions))
  
  return test_predictions
  
  
logistic_reg(data)
mlp_classifier(data)
# random_forest_classifier(data)
# imbalanced_learn(data)
# imbalanced_dataset_smote(data)
  