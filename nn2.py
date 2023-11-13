import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
data = pd.read_csv('trainingset.csv')

# Separate features and labels
X = data.iloc[:, 1:-1]  # Features
y = data['ClaimAmount']  # Labels

# Create a binary label indicating whether the claim is 0 or greater than 0
data['ClaimLabel'] = (data['ClaimAmount'] > 0).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['ClaimLabel'], test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Upsample the minority class to address imbalance
X_train_up, y_train_up = resample(X_train[y_train > 0], y_train[y_train > 0], replace=True, n_samples=len(y_train[y_train == 0]), random_state=42)

X_train_balanced = np.vstack([X_train[y_train == 0], X_train_up])
y_train_balanced = np.concatenate([y_train[y_train == 0], y_train_up])

# Define a more complex neural network model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='linear')  # Use linear activation for regression
])

# Compile the model with a lower learning rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model for binary classification
model.fit(X_train_balanced, y_train_balanced, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('trained_model.h5')

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print(f'Test loss (MSE): {test_loss}')

# Use the model to predict whether ClaimAmount is greater than 0 for the entire dataset
predictions_binary = model.predict(X)

# Use the model to predict ClaimAmount for the entire dataset
predictions_regression = model.predict(X)
data['ClaimAmountPrediction'] = np.abs(predictions_regression.flatten()) * data['ClaimAmount']  # Take the absolute value of predictions

# Save the results to a new CSV file
data[['rowIndex', 'ClaimAmountPrediction']].to_csv('output_results.csv', index=False)
