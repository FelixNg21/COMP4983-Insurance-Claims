import autokeras as ak
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Load the dataset
data = pd.read_csv('trainingset.csv')
x_data = data.iloc[:, 1:-1]  # Features
y_data = data['ClaimAmount']  # Labels

xtrain, xval, ytrain, yval = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


def auto_model():
    reg = ak.StructuredDataRegressor(overwrite=False)
    reg.fit(x_data, y_data, use_multiprocessing=True)

    predicted_y = reg.predict(xval)
    print(predicted_y)
    print(reg.evaluate(xval, yval))
    model = reg.export_model()
    model.summary()

auto_model()
