from lazypredict.Supervised import LazyClassifier, LazyRegressor
import lazypredict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from joblib import dump, load, parallel_backend, Parallel, delayed
from imblearn.over_sampling import SMOTE
from collections import Counter

# import data
data = pd.read_csv('trainingset.csv')

# Separate features and labels
x = data.iloc[:, 1:-1]  # Features
y = data['ClaimAmount']  # Labels
y_classification = (data['ClaimAmount'] > 0).astype(int)
print(Counter(y_classification))

# Split the dataset into training and testing sets (classification)
x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(x, y_classification, test_size=0.2,
                                                            random_state=42)

# Split the dataset into training and testing sets (regression)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(x, y, test_size=0.2, random_state=42)

# resample
smote = SMOTE(random_state=42)
print(Counter(y_train_c))
x_train_c, y_train_c = smote.fit_resample(x_train_c, y_train_c)
print(Counter(y_train_c))

# Standardize features
scaler = StandardScaler()
x_train_c = scaler.fit_transform(x_train_c)

# Remove the high memory classifiers from the list
highmem_classifiers = ["LabelSpreading", "LabelPropagation", "BernoulliNB", "KNeighborsClassifier",
                       "ElasticNetClassifier", "GradientBoostingClassifier",
                       "HistGradientBoostingClassifier"]
classifiers = [c for c in lazypredict.Supervised.CLASSIFIERS if c[0] not in highmem_classifiers]


def do_classifiers():
    clf = LazyClassifier(classifiers=classifiers, verbose=0, ignore_warnings=True,
                         custom_metric=None)
    models_c, predictions_c = clf.fit(x_train_c, x_test_c, y_train_c, y_test_c)
    models_c.to_csv('models_c.csv')
    predictions_c.to_csv('predictions_c.csv')
    print(models_c, predictions_c)


def do_regressors():
    # Remove the high memory regressors from the list
    highmem_regressors = ["GammaRegressor", "GaussianProcessRegressor", "KernelRidge",
                          "QuantileRegressor"]
    regressors = [reg for reg in lazypredict.Supervised.REGRESSORS if
                  reg[0] not in highmem_regressors]

    reg = LazyRegressor(regressors=regressors, verbose=1, ignore_warnings=True, custom_metric=None)
    models_r, predictions_r = reg.fit(X_train_r, X_test_r, y_train_r, y_test_r)
    print(models_r)


def perform_svc(kernel_type):
    c = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    results = []

    for penalty in c:
        model = SVC(kernel=kernel_type, C=penalty)
        cv = KFold(n_splits=10, shuffle=False)
        fold_results = []
        for train_index, test_index in cv.split(x_train_c):
            x_train, x_test = x_train_c.iloc[train_index], x_train_c.iloc[test_index]
            y_train, y_test = y_train_c.iloc[train_index], y_train_c.iloc[test_index]
            print("fitting model")
            model.fit(x_train, y_train)
            print("predicting")
            y_pred = model.predict(x_test)
            fold_results.append(np.mean(y_pred != y_test))
        results.append(np.mean(fold_results))

    plt.plot(c, results, label='Misclassification Rate')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Percentage of misclassifications')
    plt.title(f'SVM {kernel_type}: % Misclassified vs C')
    plt.legend()
    plt.show()

    min_penalty = c[results.index(min(results))]
    print("Min penalty:", min_penalty)
    model = SVC(kernel=kernel_type, C=min_penalty)
    model.fit(x_train_c, y_train_c)
    predictions = model.predict(x_test_c)
    percent_misclass = np.mean(predictions != y_test_c)
    print("Percent misclassification:", percent_misclass)

    plt.scatter(x_test_c.iloc[:, 0], x_test_c.iloc[:, 1], c=y_test_c, s=50, cmap='autumn')
    # plot_svc_decision_function(model)
    # plt.title(f'SVM with {kernel_type} kernel')
    # plt.legend(ytest.unique())
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.show()
    # plt.close()

    # save the model
    dump(model, f'{kernel_type}_SVC_model.joblib')


def train_and_evaluate(penalty, kernel_type, x_train, y_train, x_test, y_test):
    model = SVC(kernel=kernel_type, C=penalty)
    print("fitting model")
    model.fit(x_train, y_train)
    print("predicting")
    y_pred = model.predict(x_test)
    index = c.index(penalty)
    dump(model, f'{index + 6}_{kernel_type}_SVC_model.joblib')
    return np.mean(y_pred != y_test)


def parallel():
    kernel_type = 'linear'
    results = []
    cv = KFold(n_splits=2, shuffle=False)
    for train_index, test_index in cv.split(x_train_c):
        results = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate)(penalty, kernel_type, x_train_c[train_index],
                                        y_train_c[train_index], x_train_c[test_index],
                                        y_train_c[test_index]) for penalty in c)
    plt.plot(c, results, label='Misclassification Rate')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Percentage of misclassifications')
    plt.title(f'SVM {kernel_type}: % Misclassified vs C')
    plt.legend()
    plt.show()

    min_penalty = c[results.index(min(results))]
    print("Min penalty:", min_penalty)
    model = SVC(kernel=kernel_type, C=min_penalty)
    model.fit(x_train_c, y_train_c)
    predictions = model.predict(x_test_c)
    percent_misclass = np.mean(predictions != y_test_c)
    print("Percent misclassification:", percent_misclass)


c = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
# c = [100, 1000]
parallel()
# perform_svc('linear')
