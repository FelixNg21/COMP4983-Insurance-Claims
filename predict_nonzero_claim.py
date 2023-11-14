
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils


data = pd.read_csv('trainingset.csv')

# preprocess the data and split into training and test sets
# label is the column name of what you want to predict
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
  
  print("Before oversampling")
  print(Counter(train_labels))
  
  # balance the dataset using smote
  smote = SMOTE(sampling_strategy='auto')
  train_features, train_labels = smote.fit_resample(train_features, train_labels)
  
  print("\nAfter oversampling")
  print(Counter(train_labels))
  
  logistic_reg_model = LogisticRegression(max_iter=5000)
  logistic_reg_model.fit(train_features, train_labels)
  log_reg_predictions = logistic_reg_model.predict(test_features)
  
  df = pd.read_csv('non_zero_claim.csv', header=None)
  df['predictions'] = log_reg_predictions
  df.to_csv('non_zero_claim.csv', index=False, header=False)
  
  print("\nTest labels:")
  print(Counter(test_labels))
  print("\nPredicted labels:")
  print(Counter(log_reg_predictions))
  
  mean_accuracy = logistic_reg_model.score(test_features, test_labels)
  print('\nMean accuracy:', mean_accuracy)
  
  return log_reg_predictions


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
  test_features, test_labels = smote.fit_resample(test_features, test_labels)
  
  train_features, test_features = standardize_features(train_features, test_features)
  
  classifier.fit(train_features, train_labels)
  
  pickle_file_name = 'mlp_model.pkl'
  
  # save the model
  with open(pickle_file_name, 'wb') as file:
    pickle.dump(classifier, file)
  
  print("Saved the model to:", pickle_file_name)
    

def mlp_classify_test(data):
  print('\nMLP classifier')
  train_features, train_labels, test_features, test_labels = insert_binary_column(data)
  train_features = train_features.drop('ClaimAmount', axis='columns', inplace=False)
  test_features = test_features.drop('ClaimAmount', axis='columns', inplace=False)
  
  # balance the dataset using smote
  smote = SMOTE(sampling_strategy='auto')
  train_features, train_labels = smote.fit_resample(train_features, train_labels)
  test_features, test_labels = smote.fit_resample(test_features, test_labels)
  
  train_features, test_features = standardize_features(train_features, test_features)
  
  # load the model
  with open('mlp_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
  
  test_predictions = loaded_model.predict(test_features)
  
  print("\nTest labels:")
  print(Counter(test_labels))
  print("\nPredicted labels:")
  print(Counter(test_predictions))
  print("Accuracy of MLPClassifier : ", loaded_model.score(test_features, test_labels))
  
  # test_labels.to_csv("non_zero_claims", index=False, header=False)
  # df = pd.read_csv("non_zero_claims", header=None)
  # df[''] = price_pred
  # df.to_csv(csv_file_name, index=False, header=False)
  
  
def view_learning_curve(history_dict):
  
  # loss
  loss_values = history_dict['loss']
  val_loss_values = history_dict['val_loss']
  
  epochs = range(1, len(loss_values) + 1)
  
  plt.plot(epochs, loss_values, 'bo', label='Training loss')
  plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
  
def training_validation_accuracy(history_dict):
  acc = history_dict['accuracy']
  val_acc = history_dict['val_accuracy']
  
  epochs = range(1, len(acc) + 1)
  
  plt.plot(epochs, acc, 'bo', label='Training accuracy')
  plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()

# https://medium.com/luca-chuangs-bapm-notes/build-a-neural-network-in-python-binary-classification-49596d7dcabf
def nn_classifier(data):
  print('\nNeural network classifier')
  train_features, train_labels, test_features, test_labels = insert_binary_column(data)
  train_features = train_features.drop('ClaimAmount', axis='columns', inplace=False)
  test_features = test_features.drop('ClaimAmount', axis='columns', inplace=False)
  
  # balance the dataset using smote
  smote = SMOTE(sampling_strategy='auto')
  train_features, train_labels = smote.fit_resample(train_features, train_labels)
  
  train_features = np.array(train_features)
  
  model = Sequential()
  model.add(Dense(128, input_dim=train_features.shape[1], activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))
  model.summary()
  
  model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_absolute_error')
  
  # early stopping calllback will stop the training when there is no improvement in
  # the validation loss for 10 consecutive epochs
  es = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)
  
  num_epochs = 10
  history = model.fit(
    train_features,
    train_labels,
    # callbacks=[es],
    epochs=num_epochs,
    batch_size=10,
    validation_split=0.2,
    shuffle=True,
    verbose=1
  )
  
  # learning curve(loss) - training and validation loss by epoch
  view_learning_curve(history.history)
  # learning curve(accuracy) - training and validation accuracy by epoch
  training_validation_accuracy(history.history)
  
  # model.predict(test_features)
  test_predictions = np.round(model.predict(test_features), 0)
  print(confusion_matrix(test_labels, test_predictions))
  print(classification_report(test_labels, test_predictions))
    
  

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
  
  
# logistic_reg(data)
# mlp_classifier(data)
mlp_classify_test(data)
# nn_classifier(data)
# random_forest_classifier(data)
# imbalanced_learn(data)
# imbalanced_dataset_smote(data)
  