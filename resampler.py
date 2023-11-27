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

data = pd.read_csv('trainingset.csv')

X = data.iloc[:, 1:-1]  # Features
y = data['ClaimAmount']