import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier, KerasRegressor
from imblearn.over_sampling import SMOTE

def create_model(hp_config):
    optimizer = hp_config.get('optimizer', 'adam')
    activation = hp_config.get('activation', 'relu')
    dropout_rate = hp_config.get('dropout_rate', 0.5)
    layer_nodes = hp_config.get('layer_nodes', [128, 64, 32])  # List of nodes per layer

    model = keras.Sequential()
    for i, nodes in enumerate(layer_nodes):
        if i == 0:
            model.add(layers.Dense(nodes, activation=activation, input_dim=X_train.shape[1]))
        else:
            model.add(layers.Dense(nodes, activation=activation))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='linear'))  # Output layer

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    print(model.summary())
    return model


# Load the dataset
data = pd.read_csv('trainingset.csv')
# drop row index column

# Separate features and labels
X = data.iloc[:, 1:-1]  # Features
# y = data['ClaimAmount']  # Labels

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


# Instantiate SMOTE
smote = SMOTE(random_state=42)
# Resample the dataset
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)




hp_configs = [
    # {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.3, 'layer_nodes': [128, 64]},
    # {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.4, 'layer_nodes': [128, 64, 32]},
    # {'optimizer': 'SGD', 'activation': 'relu', 'dropout_rate': 0.3, 'layer_nodes': [64, 64]},
    # {'optimizer': 'SGD', 'activation': 'relu', 'dropout_rate': 0.4, 'layer_nodes': [64, 32, 16]},
    # {'optimizer': 'RMSprop', 'activation': 'relu', 'dropout_rate': 0.2, 'layer_nodes': [128, 128]},
    # {'optimizer': 'RMSprop', 'activation': 'tanh', 'dropout_rate': 0.5, 'layer_nodes': [100, 50, 25]},
    # {'optimizer': 'Adam', 'activation': 'sigmoid', 'dropout_rate': 0.3, 'layer_nodes': [128, 128, 64]},
    # {'optimizer': 'SGD', 'activation': 'sigmoid', 'dropout_rate': 0.4, 'layer_nodes': [64, 64, 32]},
    {'optimizer': 'SGD', 'activation': 'relu', 'dropout_rate': 0.5, 'layer_nodes': [128, 128, 64, 32]},
    {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.5, 'layer_nodes': [128, 128, 64, 32]},
    {'optimizer': 'RMSprop', 'activation': 'relu', 'dropout_rate': 0.5, 'layer_nodes': [128, 128, 64, 32]},
    {'optimizer': 'SGD', 'activation': 'relu', 'dropout_rate': 0.25, 'layer_nodes': [128, 128, 64, 32]},
    {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.25, 'layer_nodes': [128, 128, 64, 32]},
    {'optimizer': 'RMSprop', 'activation': 'relu', 'dropout_rate': 0.25, 'layer_nodes': [128, 128, 64, 32]},
    # {'optimizer': 'SGD', 'activation': 'relu', 'dropout_rate': 0.5, 'layer_nodes': [128,  64]},
    # {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.5, 'layer_nodes': [128,  64]},
    # {'optimizer': 'RMSprop', 'activation': 'relu', 'dropout_rate': 0.5, 'layer_nodes': [128,  64]},
    # {'optimizer': 'SGD', 'activation': 'relu', 'dropout_rate': 0.25, 'layer_nodes': [128,  64]},
    # {'optimizer': 'Adam', 'activation': 'relu', 'dropout_rate': 0.25, 'layer_nodes': [128,  64]},
    # {'optimizer': 'RMSprop', 'activation': 'relu', 'dropout_rate': 0.25, 'layer_nodes': [128,  64]},
    # Add more configurations as desired
]

best_score = 0
best_config = None
best_model = None

# Loop through each hyperparameter configuration
for config in hp_configs:
    # Wrap Keras model with the current configuration
    model = KerasClassifier(model=lambda: create_model(config), epochs=5, verbose=1)
    
    # if len(hp_configs) == 1:
    #     model.fit(X_train_balanced, y_train_balanced)
    #     best_model = model
    #     break

    # Evaluate the model using cross-validation
    score = np.mean(cross_val_score(model, X_train_balanced, y_train_balanced, cv=3))

    # Keep track of the best performing configuration
    if score > best_score:
        best_score = score
        best_config = config
        # Rebuild and fit the best model
        best_model = create_model(best_config)
        best_model.fit(X_train_balanced, y_train_balanced, epochs=5, verbose=1)

# Print best configuration
print("Best Configuration:", best_config)

# Evaluate the best model on the test set
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f'Test Loss (MSE): {test_loss}, Test Accuracy: {test_accuracy}')

# train the model on the entire dataset
X = np.array(X)
y = np.array(y).reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Instantiate SMOTE
smote = SMOTE(random_state=42)
# Resample the dataset
X_balanced, y_balanced = smote.fit_resample(X, y)

best_model.fit(X_balanced, y_balanced, epochs=20, verbose=1)

# Save the best model
best_model.save('trained_model.h5')

