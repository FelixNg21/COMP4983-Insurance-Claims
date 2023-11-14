import contextlib
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.svm import NuSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.discovery import all_estimators
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('trainingset.csv')
test_data = pd.read_csv('testset.csv')

X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# bag_reg = BaggingRegressor()
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# n_scores = cross_val_score(bag_reg, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
#
# print('MAE: %.3f (%.3f)' % (np.mean(np.abs(n_scores)), np.std(n_scores)))

def get_models():
    return [
        ('huber', HuberRegressor(max_iter=1000)),
        ('ransac', RANSACRegressor()),
        ('svr', SVR()),
        ('nusvr', NuSVR()),
        ('knn', KNeighborsRegressor()),
    ]

def evaluate_model(model, x, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    return cross_val_score(
        model,
        x,
        y,
        scoring='neg_mean_absolute_error',
        cv=cv,
        n_jobs=-1,
        error_score='raise',
    )


models = get_models()
print(len(models))
results, names = [], []
for tuple in models:
    name = tuple[0]
    model = tuple[1]
    try:
        scores = evaluate_model(model, x_train, y_train)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    except Exception as e:
        print(e)

#
# plt.boxplot(results, labels=names, showmeans=True)
# plt.show()
