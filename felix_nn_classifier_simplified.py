import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
from imblearn.over_sampling import SMOTE
import numpy as np
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import f1_score

# Load the dataset
data = pd.read_csv('trainingset.csv')

# Separate features and labels
X = data.iloc[:, 1:-1]  # Features

# Create a binary label indicating whether the claim is 0 or greater than 0
data['ClaimLabel'] = (data['ClaimAmount'] > 0).astype(int)
y = data['ClaimLabel']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate SMOTE - can test with different number of neighbors
smote = SMOTE(random_state=42)
# Resample the dataset
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

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

def use_best_params(best_params):
    # Build and fit the best model
    best_model = create_model(**best_params)
    print("Best hyperparameters:", best_params)
    best_model.fit(X_train_balanced, y_train_balanced, epochs=20, verbose=1)

    # Predict on the test set
    y_pred = best_model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Evaluate the best model using F1 score
    f1 = f1_score(y_test, y_pred_binary)
    print(f'Test F1 Score: {f1}')

    # Save the best model
    best_model.save('felix_node_permutation_trained_model.h5')

# Use the specified best_params
best_params = {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.0, 'layer_nodes': [128, 128, 64]}
use_best_params(best_params)
