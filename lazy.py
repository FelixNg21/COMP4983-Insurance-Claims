from lazypredict.Supervised import LazyClassifier, LazyRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

# import data
data = pd.read_csv('trainingset.csv')

# Separate features and labels
x = data.iloc[:, 1:-1]  # Features
y = data['ClaimAmount']  # Labels
y_classification = (data['ClaimAmount'] > 0).astype(int)


# Split the dataset into training and testing sets (classification)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(x, y_classification, test_size=0.2, random_state=42)

# Split the dataset into training and testing sets (regression)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(x, y, test_size=0.2, random_state=42)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models_c, predictions_c = clf.fit(X_train_c, X_test_c, y_train_c, y_test_c)
print(models_c)

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models_r, predictions_r = reg.fit(X_train_r, X_test_r, y_train_r, y_test_r)
print(models_r)
