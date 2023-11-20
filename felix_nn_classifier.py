import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
from scikeras.wrappers import KerasClassifier
from imblearn.over_sampling import SMOTE
from itertools import product
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def create_model(optimizer='adam', activation=None, dropout_rate=0.5, layer_nodes=None):
    model = keras.Sequential()
    for i, nodes in enumerate(layer_nodes):
        if i == 0:
            model.add(layers.Dense(nodes, activation=activation, input_dim=X_train.shape[1]))
        else:
            model.add(layers.Dense(nodes, activation=activation))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'])
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
        'dropout_rate': [0.0],
        'layer_nodes': [[18, 17, 16], [18, 17, 15], [18, 17, 14], [18, 17, 13], [18, 17, 12], [18, 17, 11], [18, 17, 10], [18, 17, 9], [18, 17, 8], [18, 17, 7], [18, 17, 6], [18, 17, 5], [18, 17, 4], [18, 17, 3], [18, 17, 2], [18, 16, 15], [18, 16, 14], [18, 16, 13], [18, 16, 12], [18, 16, 11], [18, 16, 10], [18, 16, 9], [18, 16, 8], [18, 16, 7], [18, 16, 6], [18, 16, 5], [18, 16, 4], [18, 16, 3], [18, 16, 2], [18, 15, 14], [18, 15, 13], [18, 15, 12], [18, 15, 11], [18, 15, 10], [18, 15, 9], [18, 15, 8], [18, 15, 7], [18, 15, 6], [18, 15, 5], [18, 15, 4], [18, 15, 3], [18, 15, 2], [18, 14, 13], [18, 14, 12], [18, 14, 11], [18, 14, 10], [18, 14, 9], [18, 14, 8], [18, 14, 7], [18, 14, 6], [18, 14, 5], [18, 14, 4], [18, 14, 3], [18, 14, 2], [18, 13, 12], [18, 13, 11], [18, 13, 10], [18, 13, 9], [18, 13, 8], [18, 13, 7], [18, 13, 6], [18, 13, 5], [18, 13, 4], [18, 13, 3], [18, 13, 2], [18, 12, 11], [18, 12, 10], [18, 12, 9], [18, 12, 8], [18, 12, 7], [18, 12, 6], [18, 12, 5], [18, 12, 4], [18, 12, 3], [18, 12, 2], [18, 11, 10], [18, 11, 9], [18, 11, 8], [18, 11, 7], [18, 11, 6], [18, 11, 5], [18, 11, 4], [18, 11, 3], [18, 11, 2], [18, 10, 9], [18, 10, 8], [18, 10, 7], [18, 10, 6], [18, 10, 5], [18, 10, 4], [18, 10, 3], [18, 10, 2], [18, 9, 8], [18, 9, 7], [18, 9, 6], [18, 9, 5], [18, 9, 4], [18, 9, 3], [18, 9, 2], [18, 8, 7], [18, 8, 6], [18, 8, 5], [18, 8, 4], [18, 8, 3], [18, 8, 2], [18, 7, 6], [18, 7, 5], [18, 7, 4], [18, 7, 3], [18, 7, 2], [18, 6, 5], [18, 6, 4], [18, 6, 3], [18, 6, 2], [18, 5, 4], [18, 5, 3], [18, 5, 2], [18, 4, 3], [18, 4, 2], [18, 3, 2], [17, 16, 15], [17, 16, 14], [17, 16, 13], [17, 16, 12], [17, 16, 11], [17, 16, 10], [17, 16, 9], [17, 16, 8], [17, 16, 7], [17, 16, 6], [17, 16, 5], [17, 16, 4], [17, 16, 3], [17, 16, 2], [17, 15, 14], [17, 15, 13], [17, 15, 12], [17, 15, 11], [17, 15, 10], [17, 15, 9], [17, 15, 8], [17, 15, 7], [17, 15, 6], [17, 15, 5], [17, 15, 4], [17, 15, 3], [17, 15, 2], [17, 14, 13], [17, 14, 12], [17, 14, 11], [17, 14, 10], [17, 14, 9], [17, 14, 8], [17, 14, 7], [17, 14, 6], [17, 14, 5], [17, 14, 4], [17, 14, 3], [17, 14, 2], [17, 13, 12], [17, 13, 11], [17, 13, 10], [17, 13, 9], [17, 13, 8], [17, 13, 7], [17, 13, 6], [17, 13, 5], [17, 13, 4], [17, 13, 3], [17, 13, 2], [17, 12, 11], [17, 12, 10], [17, 12, 9], [17, 12, 8], [17, 12, 7], [17, 12, 6], [17, 12, 5], [17, 12, 4], [17, 12, 3], [17, 12, 2], [17, 11, 10], [17, 11, 9], [17, 11, 8], [17, 11, 7], [17, 11, 6], [17, 11, 5], [17, 11, 4], [17, 11, 3], [17, 11, 2], [17, 10, 9], [17, 10, 8], [17, 10, 7], [17, 10, 6], [17, 10, 5], [17, 10, 4], [17, 10, 3], [17, 10, 2], [17, 9, 8], [17, 9, 7], [17, 9, 6], [17, 9, 5], [17, 9, 4], [17, 9, 3], [17, 9, 2], [17, 8, 7], [17, 8, 6], [17, 8, 5], [17, 8, 4], [17, 8, 3], [17, 8, 2], [17, 7, 6], [17, 7, 5], [17, 7, 4], [17, 7, 3], [17, 7, 2], [17, 6, 5], [17, 6, 4], [17, 6, 3], [17, 6, 2], [17, 5, 4], [17, 5, 3], [17, 5, 2], [17, 4, 3], [17, 4, 2], [17, 3, 2], [16, 15, 14], [16, 15, 13], [16, 15, 12], [16, 15, 11], [16, 15, 10], [16, 15, 9], [16, 15, 8], [16, 15, 7], [16, 15, 6], [16, 15, 5], [16, 15, 4], [16, 15, 3], [16, 15, 2], [16, 14, 13], [16, 14, 12], [16, 14, 11], [16, 14, 10], [16, 14, 9], [16, 14, 8], [16, 14, 7], [16, 14, 6], [16, 14, 5], [16, 14, 4], [16, 14, 3], [16, 14, 2], [16, 13, 12], [16, 13, 11], [16, 13, 10], [16, 13, 9], [16, 13, 8], [16, 13, 7], [16, 13, 6], [16, 13, 5], [16, 13, 4], [16, 13, 3], [16, 13, 2], [16, 12, 11], [16, 12, 10], [16, 12, 9], [16, 12, 8], [16, 12, 7], [16, 12, 6], [16, 12, 5], [16, 12, 4], [16, 12, 3], [16, 12, 2], [16, 11, 10], [16, 11, 9], [16, 11, 8], [16, 11, 7], [16, 11, 6], [16, 11, 5], [16, 11, 4], [16, 11, 3], [16, 11, 2], [16, 10, 9], [16, 10, 8], [16, 10, 7], [16, 10, 6], [16, 10, 5], [16, 10, 4], [16, 10, 3], [16, 10, 2], [16, 9, 8], [16, 9, 7], [16, 9, 6], [16, 9, 5], [16, 9, 4], [16, 9, 3], [16, 9, 2], [16, 8, 7], [16, 8, 6], [16, 8, 5], [16, 8, 4], [16, 8, 3], [16, 8, 2], [16, 7, 6], [16, 7, 5], [16, 7, 4], [16, 7, 3], [16, 7, 2], [16, 6, 5], [16, 6, 4], [16, 6, 3], [16, 6, 2], [16, 5, 4], [16, 5, 3], [16, 5, 2], [16, 4, 3], [16, 4, 2], [16, 3, 2], [15, 14, 13], [15, 14, 12], [15, 14, 11], [15, 14, 10], [15, 14, 9], [15, 14, 8], [15, 14, 7], [15, 14, 6], [15, 14, 5], [15, 14, 4], [15, 14, 3], [15, 14, 2], [15, 13, 12], [15, 13, 11], [15, 13, 10], [15, 13, 9], [15, 13, 8], [15, 13, 7], [15, 13, 6], [15, 13, 5], [15, 13, 4], [15, 13, 3], [15, 13, 2], [15, 12, 11], [15, 12, 10], [15, 12, 9], [15, 12, 8], [15, 12, 7], [15, 12, 6], [15, 12, 5], [15, 12, 4], [15, 12, 3], [15, 12, 2], [15, 11, 10], [15, 11, 9], [15, 11, 8], [15, 11, 7], [15, 11, 6], [15, 11, 5], [15, 11, 4], [15, 11, 3], [15, 11, 2], [15, 10, 9], [15, 10, 8], [15, 10, 7], [15, 10, 6], [15, 10, 5], [15, 10, 4], [15, 10, 3], [15, 10, 2], [15, 9, 8], [15, 9, 7], [15, 9, 6], [15, 9, 5], [15, 9, 4], [15, 9, 3], [15, 9, 2], [15, 8, 7], [15, 8, 6], [15, 8, 5], [15, 8, 4], [15, 8, 3], [15, 8, 2], [15, 7, 6], [15, 7, 5], [15, 7, 4], [15, 7, 3], [15, 7, 2], [15, 6, 5], [15, 6, 4], [15, 6, 3], [15, 6, 2], [15, 5, 4], [15, 5, 3], [15, 5, 2], [15, 4, 3], [15, 4, 2], [15, 3, 2], [14, 13, 12], [14, 13, 11], [14, 13, 10], [14, 13, 9], [14, 13, 8], [14, 13, 7], [14, 13, 6], [14, 13, 5], [14, 13, 4], [14, 13, 3], [14, 13, 2], [14, 12, 11], [14, 12, 10], [14, 12, 9], [14, 12, 8], [14, 12, 7], [14, 12, 6], [14, 12, 5], [14, 12, 4], [14, 12, 3], [14, 12, 2], [14, 11, 10], [14, 11, 9], [14, 11, 8], [14, 11, 7], [14, 11, 6], [14, 11, 5], [14, 11, 4], [14, 11, 3], [14, 11, 2], [14, 10, 9], [14, 10, 8], [14, 10, 7], [14, 10, 6], [14, 10, 5], [14, 10, 4], [14, 10, 3], [14, 10, 2], [14, 9, 8], [14, 9, 7], [14, 9, 6], [14, 9, 5], [14, 9, 4], [14, 9, 3], [14, 9, 2], [14, 8, 7], [14, 8, 6], [14, 8, 5], [14, 8, 4], [14, 8, 3], [14, 8, 2], [14, 7, 6], [14, 7, 5], [14, 7, 4], [14, 7, 3], [14, 7, 2], [14, 6, 5], [14, 6, 4], [14, 6, 3], [14, 6, 2], [14, 5, 4], [14, 5, 3], [14, 5, 2], [14, 4, 3], [14, 4, 2], [14, 3, 2], [13, 12, 11], [13, 12, 10], [13, 12, 9], [13, 12, 8], [13, 12, 7], [13, 12, 6], [13, 12, 5], [13, 12, 4], [13, 12, 3], [13, 12, 2], [13, 11, 10], [13, 11, 9], [13, 11, 8], [13, 11, 7], [13, 11, 6], [13, 11, 5], [13, 11, 4], [13, 11, 3], [13, 11, 2], [13, 10, 9], [13, 10, 8], [13, 10, 7], [13, 10, 6], [13, 10, 5], [13, 10, 4], [13, 10, 3], [13, 10, 2], [13, 9, 8], [13, 9, 7], [13, 9, 6], [13, 9, 5], [13, 9, 4], [13, 9, 3], [13, 9, 2], [13, 8, 7], [13, 8, 6], [13, 8, 5], [13, 8, 4], [13, 8, 3], [13, 8, 2], [13, 7, 6], [13, 7, 5], [13, 7, 4], [13, 7, 3], [13, 7, 2], [13, 6, 5], [13, 6, 4], [13, 6, 3], [13, 6, 2], [13, 5, 4], [13, 5, 3], [13, 5, 2], [13, 4, 3], [13, 4, 2], [13, 3, 2], [12, 11, 10], [12, 11, 9], [12, 11, 8], [12, 11, 7], [12, 11, 6], [12, 11, 5], [12, 11, 4], [12, 11, 3], [12, 11, 2], [12, 10, 9], [12, 10, 8], [12, 10, 7], [12, 10, 6], [12, 10, 5], [12, 10, 4], [12, 10, 3], [12, 10, 2], [12, 9, 8], [12, 9, 7], [12, 9, 6], [12, 9, 5], [12, 9, 4], [12, 9, 3], [12, 9, 2], [12, 8, 7], [12, 8, 6], [12, 8, 5], [12, 8, 4], [12, 8, 3], [12, 8, 2], [12, 7, 6], [12, 7, 5], [12, 7, 4], [12, 7, 3], [12, 7, 2], [12, 6, 5], [12, 6, 4], [12, 6, 3], [12, 6, 2], [12, 5, 4], [12, 5, 3], [12, 5, 2], [12, 4, 3], [12, 4, 2], [12, 3, 2], [11, 10, 9], [11, 10, 8], [11, 10, 7], [11, 10, 6], [11, 10, 5], [11, 10, 4], [11, 10, 3], [11, 10, 2], [11, 9, 8], [11, 9, 7], [11, 9, 6], [11, 9, 5], [11, 9, 4], [11, 9, 3], [11, 9, 2], [11, 8, 7], [11, 8, 6], [11, 8, 5], [11, 8, 4], [11, 8, 3], [11, 8, 2], [11, 7, 6], [11, 7, 5], [11, 7, 4], [11, 7, 3], [11, 7, 2], [11, 6, 5], [11, 6, 4], [11, 6, 3], [11, 6, 2], [11, 5, 4], [11, 5, 3], [11, 5, 2], [11, 4, 3], [11, 4, 2], [11, 3, 2], [10, 9, 8], [10, 9, 7], [10, 9, 6], [10, 9, 5], [10, 9, 4], [10, 9, 3], [10, 9, 2], [10, 8, 7], [10, 8, 6], [10, 8, 5], [10, 8, 4], [10, 8, 3], [10, 8, 2], [10, 7, 6], [10, 7, 5], [10, 7, 4], [10, 7, 3], [10, 7, 2], [10, 6, 5], [10, 6, 4], [10, 6, 3], [10, 6, 2], [10, 5, 4], [10, 5, 3], [10, 5, 2], [10, 4, 3], [10, 4, 2], [10, 3, 2], [9, 8, 7], [9, 8, 6], [9, 8, 5], [9, 8, 4], [9, 8, 3], [9, 8, 2], [9, 7, 6], [9, 7, 5], [9, 7, 4], [9, 7, 3], [9, 7, 2], [9, 6, 5], [9, 6, 4], [9, 6, 3], [9, 6, 2], [9, 5, 4], [9, 5, 3], [9, 5, 2], [9, 4, 3], [9, 4, 2], [9, 3, 2], [8, 7, 6], [8, 7, 5], [8, 7, 4], [8, 7, 3], [8, 7, 2], [8, 6, 5], [8, 6, 4], [8, 6, 3], [8, 6, 2], [8, 5, 4], [8, 5, 3], [8, 5, 2], [8, 4, 3], [8, 4, 2], [8, 3, 2], [7, 6, 5], [7, 6, 4], [7, 6, 3], [7, 6, 2], [7, 5, 4], [7, 5, 3], [7, 5, 2], [7, 4, 3], [7, 4, 2], [7, 3, 2], [6, 5, 4], [6, 5, 3], [6, 5, 2], [6, 4, 3], [6, 4, 2], [6, 3, 2], [5, 4, 3], [5, 4, 2], [5, 3, 2], [4, 3, 2]]
    }

    best_score = 0
    best_params = None
    best_model = None

    # Loop through each hyperparameter combination
    for params in product(*param_grid.values()):
        hyperparameters = dict(zip(param_grid.keys(), params))
        print(hyperparameters)
        model = KerasClassifier(build_fn=create_model, epochs=5, verbose=0, **hyperparameters)
        score = np.mean(cross_val_score(model, X_train_balanced, y_train_balanced, cv=3, scoring='f1'))
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
    print(f'Test Loss (Binary Crossentropy): {test_loss}, Test Accuracy: {test_accuracy}, Test Binary Accuracy: {test_binary_accuracy}')

    # Save the best model
    best_model.save('felix_node_permutation_trained_model.h5')

best_params= find_best_hyperparameters()
# best_params =  {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.0, 'layer_nodes': [128, 128, 64]} #Test Loss (Binary Crossentropy): 0.3443473279476166, Test Accuracy: 0.9223571419715881, Test Binary Accuracy: 0.9223571419715881
#best_params={'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.0, 'layer_nodes': [128, 128, 64, 32]} #Loss (Binary Crossentropy): 0.40981221199035645, Test Accuracy: 0.9097142815589905, Test Binary Accuracy: 0.9097142815589905
# best_params = {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.0, 'layer_nodes': [128, 128, 64, 64]} #Test Loss (Binary Crossentropy): 0.4488109052181244, Test Accuracy: 0.920285701751709, Test Binary Accuracy: 0.920285701751709
use_best_params(best_params)

# the first run of this code experimented with different optimizers, activation functions, and dropout rates along with number of nodes and layers
# it appears that Adam is the best optimizer, relu is the best activation function, and 0.0 is the best dropout rate
# in subsequent runs, different layer nodes were tested and it appears that 128, 128, 64, 64 is the best configuration of the hidden layer
# upon further research, the number of nodes in the hidden layer should be between the number of input nodes and output nodes
# since the number of input nodes is the number of features, which is 18, subsequent runs will experiment with permutations