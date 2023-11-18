import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter

from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import os

print(tf.__version__)

# Load the dataset
data = pd.read_csv('trainingset.csv')
x_data = data.iloc[:, 1:-1]  # Features
y_data = data.iloc[:, -1]  # Labels
y_data_c = (y_data > 0).astype(int)

# Standardize features
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data_c, test_size=0.2,
                                                    random_state=42)

tf.random.set_seed(42)


def first_model():
    model_1 = tf.keras.Sequential([tf.keras.layers.Dense(1),
                                   tf.keras.layers.Dense(100, activation='relu'),
                                   tf.keras.layers.Dense(10, activation='relu'),
                                   tf.keras.layers.Dense(1, activation='sigmoid')])
    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
                    metrics=['accuracy'])
    model_1.fit(x_train, y_train, epochs=100)
    loss, accuracy = model_1.evaluate(x_test, y_test)
    print(Counter(y_test))
    print(f' Model loss on the test set: {loss}')
    print(f' Model accuracy on the test set: {100 * accuracy}')


METRICS = [
    keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
    keras.metrics.MeanSquaredError(name='Brier score'),
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Dense(
            16, activation='relu',
            input_shape=(x_train.shape[-1],)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid',
                           bias_initializer=output_bias),
    ])

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

num_zeros = int(np.sum(y_train == 0))
num_ones = int(np.sum(y_train == 1))

initial_bias = np.log([num_ones/num_zeros])
print(initial_bias)

model = make_model(output_bias=initial_bias)
model.summary()
print(model.predict(x_train))

results = model.evaluate(x_train, y_train, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0])) # before setting bias: 0.7906, after: 0.1996

initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)

model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(x_test, y_test),
    verbose=0)

model = make_model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(x_test, y_test),
    verbose=0)

def plot_loss(history, label, n):
  # Use a log scale on y-axis to show the wide range of values.
  plt.semilogy(history.epoch, history.history['loss'],
                label='Train ' + label)
  plt.semilogy(history.epoch, history.history['val_loss'],
                label='Val ' + label,
               linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()


plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)
plt.show()
