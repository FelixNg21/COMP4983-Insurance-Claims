import keras.models
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from keras import layers
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('trainingset.csv')
# drop row index column

# Separate features and labels
X = data.iloc[:, 1:-1]  # Features

# Create a binary label indicating whether the claim is 0 or greater than 0
data['ClaimLabel'] = (data['ClaimAmount'] > 0).astype(int)
y = data['ClaimLabel']

# scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print number of features in the training set
print(f"Number of features in X_train: {X_train.shape[1]}")

# Define the base classifier
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
base_classifier = KerasClassifier(build_fn=create_model, verbose=1, batch_size=16, epochs=20, **best_params)
# base_classifier = RandomForestClassifier(random_state=42)

# Define oversampling techniques
oversamplers = {
    'Random Oversampling': RandomOverSampler(sampling_strategy='auto', random_state=42),
    'SMOTE': SMOTE(sampling_strategy='auto', random_state=42),
    'ADASYN': ADASYN(sampling_strategy='auto', random_state=42)
}

# Evaluate each oversampling technique using cross-validation
results = {}
scoring_metric = 'f1'  # Choose the metric you want to optimize (e.g., 'precision', 'recall', 'f1', 'accuracy')

for oversampler_name, oversampler in oversamplers.items():
    print("Doing cross validation for", oversampler_name)
    pipeline = make_pipeline(oversampler, base_classifier)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring_metric, verbose=1, n_jobs=-1) #TODO change njobs
    results[oversampler_name] = scores

# Print average performance for each oversampling technique
for oversampler_name, scores in results.items():
    print(f'{oversampler_name}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})')

# Select the oversampler with the best average performance
best_oversampler_name = max(results, key=lambda k: np.mean(results[k]))
best_oversampler = oversamplers[best_oversampler_name]
print(best_oversampler)

# Now, you can use the best oversampler in your final model training
final_pipeline = make_pipeline(best_oversampler, base_classifier)
final_pipeline.fit(X, y)
