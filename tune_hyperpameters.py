import keras
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Load your dataset
# Replace 'your_data.csv' with the path to your dataset
data = pd.read_csv('trainingset.csv')

# Assume the target variable is named 'target'
X = data.iloc[:, 1:-1]

# Create a binary label indicating whether the claim is 0 or greater than 0
data['ClaimLabel'] = (data['ClaimAmount'] > 0).astype(int)
y = data['ClaimLabel']

# scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
best_params = {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.0, 'layer_nodes': [12, 8]}
clf = KerasClassifier(build_fn=create_model, verbose=1, batch_size=16, epochs=20, **best_params)

# Define the SMOTE object
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Define the pipeline with SMOTE and the classifier
pipeline = Pipeline(steps=[('smote', smote), ('classifier', clf)])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'smote__k_neighbors': [3, 5, 7],  # Adjust the range of k_neighbors as needed
    'smote__sampling_strategy': ['auto', 0.5, 0.75, 1.0, 1.1, 1.2]  # Adjust the sampling_strategy values as needed
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Use the best model to make predictions on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the performance of the best model
f1 = f1_score(y_test, y_pred)
print("Test Accuracy:", f1)
# Best Hyperparameters: {'smote__k_neighbors': 5, 'smote__sampling_strategy': 0.5}
# Test Accuracy: 0.15221238938053097