import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from imblearn.over_sampling import SMOTE
from itertools import product
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


# Set seeds for reproducibility
np.random.seed(42)


import joblib
warnings.filterwarnings('ignore')


def create_model(optimizer='adam', activation='relu', dropout_rate=0.5, layer_nodes=None):
    model = keras.Sequential()
    for i, nodes in enumerate(layer_nodes):
        if i == 0:
            model.add(layers.Dense(nodes, activation=activation, input_dim=X_train_balanced.shape[1]))
        else:
            model.add(layers.Dense(nodes, activation=activation))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer

    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_accuracy'])
    return model


# Load the dataset
data = pd.read_csv('trainingset.csv')
# drop row index column

# Separate features and labels
X = data.iloc[:, 1:-1]  # Features

# Create a binary label indicating whether the claim is 0 or greater than 0
data['ClaimLabel'] = (data['ClaimAmount'] > 0).astype(int)
y = data['ClaimLabel']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print number of features in the training set
print(f"Number of features in X_train: {X_train.shape[1]}")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'felix_nn_classifier_scaler.joblib')
print("DUMPED")

# Instantiate SMOTE - can test with different number of neighbors
smote = SMOTE(random_state=42, k_neighbors=5, n_jobs=-1, sampling_strategy=0.6)
# Resample the dataset
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Feature Selection with RFE
# rf_classifier_for_rfe = RandomForestClassifier(random_state=42)
# rfe = RFE(estimator=rf_classifier_for_rfe, n_features_to_select=10, verbose=2)  # Adjust the number of features
# X_train_balanced = rfe.fit_transform(X_train_balanced, y_train_balanced)
# X_test = rfe.transform(X_test)

# print(f"Number of features in X_train_balanced: {X_train_balanced.shape[1]}")
# print("Feature Ranking: ", rfe.ranking_)
# print("Features Selected: ", rfe.support_)

# Feature ranking array
feature_ranking = [1, 1, 1, 6, 7, 2, 1, 1, 9, 1, 3, 1, 5, 8, 1, 1, 1, 4]

# Select indices of important features (ranking of 1)
important_features_indices = [i for i, rank in enumerate(feature_ranking) if rank < 7]

# Select important features from datasets
X_train_balanced = X_train_balanced[:, important_features_indices]
X_test = X_test[:, important_features_indices]


def find_best_hyperparameters():
    param_grid = {
        'optimizer': ['Adam'],
        'activation': ['relu'],
        'dropout_rate': [0.0, 0.1],
        'layer_nodes': [[12, 8], [128, 128, 64]]
    }  # old layer nodes: [18, 17, 16], [18, 17, 15], [18, 17, 14], [18, 17, 13], [18, 17, 12], [18, 17, 11], [18, 17, 10], [18, 17, 9], [18, 17, 8], [18, 17, 7], [18, 17, 6], [18, 17, 5], [18, 17, 4], [18, 17, 3], [18, 17, 2], [18, 16, 15], [18, 16, 14], [18, 16, 13], [18, 16, 12], [18, 16, 11], [18, 16, 10], [18, 16, 9], [18, 16, 8], [18, 16, 7], [18, 16, 6], [18, 16, 5], [18, 16, 4], [18, 16, 3], [18, 16, 2],
    # old: [[12, 11], [12, 10], [12, 9], [12, 8], [12, 7], [12, 6], [12, 5],
    #                         [12, 4], [12, 3], [12, 2], [11, 10], [11, 10], [11, 9], [11, 8], [11, 7],
    #                         [11, 6], [11, 5], [11, 4], [11, 3], [11, 2], [10, 9], [10, 8], [10, 7],
    #                         [10, 6], [10, 5], [10, 4], [10, 3], [9, 8], [9, 7], [9, 6], [9, 5], [9, 4],
    #                         [9, 3], [9, 2],[8, 7], [8, 6], [8, 5], [8, 4], [8, 3, ], [8, 2], [7, 6], [7, 5], [7, 4],
    #                         [7, 3], [7, 2], [6, 5], [6, 4], [6, 3], [6, 2], [5, 4], [5, 3], [5, 2], [4, 3], [4, 2],
    #                         [3, 2]]
    best_score = 0
    best_params = None
    early_stopping = EarlyStopping(monitor='binary_accuracy', patience=1)

    # Loop through each hyperparameter combination
    for params in product(*param_grid.values()):
        hyperparameters = dict(zip(param_grid.keys(), params))
        print(hyperparameters)
        model = KerasClassifier(build_fn=create_model, epochs=5, verbose=1,
                                callbacks=[early_stopping], **hyperparameters)
        score = np.mean(
            cross_val_score(model, X_train_balanced, y_train_balanced, cv=3, scoring='f1'))
        print(score)
        if score > best_score:
            best_score = score
            best_params = hyperparameters
    return best_params


def use_best_params(best_params):
    # Build and fit the best model
    best_model = create_model(**best_params)
    print("Best hyperparameters:", best_params)
    best_model.fit(X_train_balanced, y_train_balanced, epochs=20, verbose=1)

    # Predictions for evaluation
    y_pred_train = best_model.predict(X_train_balanced)
    y_pred_test = best_model.predict(X_test)

    # Evaluation metrics for training set
    train_accuracy = accuracy_score(y_train_balanced, y_pred_train.round())
    train_precision = precision_score(y_train_balanced, y_pred_train.round())
    train_recall = recall_score(y_train_balanced, y_pred_train.round())
    train_f1 = f1_score(y_train_balanced, y_pred_train.round())

    print(f"Training Metrics: Accuracy: {train_accuracy}, Precision: {train_precision}, Recall: {train_recall}, F1 Score: {train_f1}")

    # Evaluation metrics for test set
    test_accuracy = accuracy_score(y_test, y_pred_test.round())
    test_precision = precision_score(y_test, y_pred_test.round())
    test_recall = recall_score(y_test, y_pred_test.round())
    test_f1 = f1_score(y_test, y_pred_test.round())

    print(f"Test Metrics: Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1}")

    # Combine training and test data
    X_full = np.concatenate((X_train_balanced, X_test), axis=0)
    y_full = np.concatenate((y_train_balanced, y_test), axis=0)

    # Train the model on the full dataset
    best_model.fit(X_full, y_full, epochs=20, verbose=1)

    # Save the best model
    best_model.save('felix_node_permutation_trained_model.h5')


best_params = find_best_hyperparameters()
# best_params = {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.0, 'layer_nodes': [128, 128, 64]} # KNeighbour = 4 Test Loss (Binary Crossentropy): 0.3418712317943573, Test Accuracy: 0.926714301109314, Test Binary Accuracy: 0.926714301109314
# best_params =  {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.0, 'layer_nodes': [128, 128, 64]} #Test Loss (Binary Crossentropy): 0.3443473279476166, Test Accuracy: 0.9223571419715881, Test Binary Accuracy: 0.9223571419715881
# best_params={'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.0, 'layer_nodes': [128, 128, 64, 32]} #Loss (Binary Crossentropy): 0.40981221199035645, Test Accuracy: 0.9097142815589905, Test Binary Accuracy: 0.9097142815589905
# best_params = {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.0, 'layer_nodes': [128, 128, 64, 64]} #Test Loss (Binary Crossentropy): 0.4488109052181244, Test Accuracy: 0.920285701751709, Test Binary Accuracy: 0.920285701751709
# Best hyperparameters: {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.0, 'layer_nodes': [11, 3]} # Test Loss (Binary Crossentropy): 0.7031096816062927, Test Accuracy: 0.607785701751709, Test Binary Accuracy: 0.607785701751709
use_best_params(best_params)

# the first run of this code experimented with different optimizers, activation functions, and dropout rates along with number of nodes and layers
# it appears that Adam is the best optimizer, relu is the best activation function, and 0.0 is the best dropout rate
# in subsequent runs, different layer nodes were tested and it appears that 128, 128, 64, 64 is the best configuration of the hidden layer
# upon further research, the number of nodes in the hidden layer should be between the number of input nodes and output nodes
# since the number of input nodes is the number of features, which is 18, subsequent runs will experiment with permutations
