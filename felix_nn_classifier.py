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

warnings.filterwarnings('ignore')


def create_model(optimizer='adam', activation='relu', dropout_rate=0.5, layer_nodes=None):
    model = keras.Sequential()
    for i, nodes in enumerate(layer_nodes):
        if i == 0:
            model.add(layers.Dense(nodes, activation=activation, input_dim=X_train.shape[1]))
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

# Instantiate SMOTE - can test with different number of neighbors
smote = SMOTE(random_state=42)
# Resample the dataset
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


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

    # Evaluate the best model on the test set
    test_loss, test_accuracy, test_binary_accuracy = best_model.evaluate(X_test, y_test)
    print(
        f'Test Loss (Binary Crossentropy): {test_loss}, Test Accuracy: {test_accuracy}, Test Binary Accuracy: {test_binary_accuracy}')

    # Save the best model
    best_model.save('felix_node_permutation_trained_model.h5')


best_params = find_best_hyperparameters()
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