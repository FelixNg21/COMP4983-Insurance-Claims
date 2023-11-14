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
    all_regs = []
    estimators = all_estimators(type_filter='regressor')
    for name, RegClass in estimators:
        with contextlib.suppress(Exception):
            if name != 'BaggingRegressor':
                reg = RegClass()
                all_regs.append((name, reg))
    return all_regs


# def get_models():
#     n_trees = [10, 50, 100, 500, 1000, 5000]
#     return {str(i): BaggingRegressor(n_estimators=i) for i in n_trees}
# 1000 trees is good

# def get_models():
#     models = dict()
#     for i in np.arange(1, 10, 1):
#         key = '%.1f' % i
#         models[key] = BaggingRegressor(max_samples=i)
#     return models

# def get_models(): # 1 neighbour is good
#     models = {}
#     for i in range(1, 2):
#         base = KNeighborsRegressor(n_neighbors=i)
#         models[str(i)] = BaggingRegressor(estimator=base, n_estimators=500)
#     return models

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
