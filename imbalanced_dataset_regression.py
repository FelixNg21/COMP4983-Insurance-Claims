import pandas as pd
import numpy as np
import resreg
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import itertools

import warnings

warnings.filterwarnings('ignore')


# https://github.com/jafetgado/resreg

class Imbalanced_Dataset_Reg:
    """
    Initialize the ImbalancedDatasetRegression class.

    Args:
        path (str): The path to the CSV file containing the dataset.
        col (str): The name of the label column.
        threshold (float): The threshold value for creating binary values. (between high and low)
        performance (bool): Whether to plot performance or not.

    Attributes:
        data (pandas.DataFrame): The loaded dataset.
        X (pandas.DataFrame): The feature matrix.
        y (pandas.Series): The target variable.
        bins (list): The list of threshold values.
        cache (dict): The dictionary for storing performance results.
        ticks_font (dict): The dictionary for setting the font size of the ticks.
        label_font (dict): The dictionary for setting the font size of the labels.
        title_font (dict): The dictionary for setting the font size of the title.
        performance (bool): Whether to plot performance or not.

    """

    def __init__(self, path, col, threshold, performance=False):
        self.data = pd.read_csv(path)
        self.X = self.data.drop(col, axis='columns', inplace=False)
        self.y = self.data.loc[:, col]
        # np.random.seed(seed=0)
        # sample = np.random.choice(range(len(self.y)), 500)
        # self.X, self.y = self.X.loc[sample, :], self.y[sample]
        self.bins = [threshold]
        self.cache = {}
        self.ticks_font = {'size': '12'}
        self.label_font = {'size': '14'}
        self.title_font = {'size': '16'}
        self.performance = performance

    def implementML(self, X_train, y_train, X_test, y_test, reg, over=None, k=None):
        reg.fit(X_train, y_train)  # fit regressor
        y_pred = reg.predict(X_test)

        # if over is not None and k is not None:
        #     df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        #     df.to_csv(f'{over}_{k}_output.csv')

        cv_err = np.mean(np.abs(y_test - y_pred))
        train_err = np.mean(np.abs(y_train - reg.predict(X_train)))
        return cv_err, train_err

    def plotPerformanceMAE(self, cv_err, train_err, title, k=5):
        plt.plot(list(range(k)), cv_err, color='royalblue', linewidth=1, label='CV error')
        plt.plot(list(range(k)), train_err, color='crimson', linewidth=1, label='Train error')
        plt.ylabel('Mean Absolute Error (MAE)', self.label_font)
        plt.xlabel('Lambda', self.label_font)
        plt.title(title+" MAE vs Neighbours", self.title_font)
        plt.legend()
        plt.show()
        plt.close()

    def plotPerformanceMSE(self, msebin, msebinerr, f1, r2, title):
        plt.bar(range(2), msebin, yerr=msebinerr, width=0.4, capsize=3, color='royalblue',
                linewidth=1, edgecolor='black')
        plt.xlim(-0.5, len(self.bins) + 0.5)
        plt.xticks(range(2), ['< {0}'.format(self.bins[0]), '≥ {0}'.format(self.bins[0])],
                   **self.ticks_font)
        plt.yticks(**self.ticks_font)
        plt.ylabel('Mean Squared Error (MSE)', self.label_font)
        plt.xlabel('Target value range', self.label_font)
        title = title + '\nf1={0}, r2={1}'.format(round(f1, 4), round(r2, 4))
        plt.title(title, self.title_font)
        plt.show()
        plt.close()

    def no_resampling(self):
        # Empty list for storing results
        cv_errs, train_errs = [], []

        # Fivefold cross validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)

        for train_index, test_index in kfold.split(self.X):
            X_train, y_train = self.X.iloc[train_index, :], self.y.iloc[train_index]
            X_test, y_test = self.X.iloc[test_index, :], self.y.iloc[test_index]
            reg = RandomForestRegressor(n_estimators=50, max_features=0.5, n_jobs=-1,
                                        random_state=0)
            cv_err, train_err = self.implementML(X_train, y_train, X_test, y_test, reg)  # Fit regressor and evaluate performance

            cv_errs.append(cv_err)
            train_errs.append(train_err)

        # Average performance
        min_index = np.argmin(cv_errs)
        min_cv_err = cv_errs[min_index]

        # View performance
        self.plotPerformanceMAE(cv_errs, train_errs, k=5, title='No resampling (None)')

        # Save performance results
        self.cache['None'] = [min_index, min_cv_err]

    def smoter(self):
        # Parameters
        overs = ['balance', 'average', 'extreme']
        ks = [x for x in range(1, 10) if x % 2]  # nearest neighbors
        params = list(itertools.product(overs, ks))
        cv_errs_store, train_errs_store = [], []

        # Grid search
        for over, k in params:
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            cv_errs, train_errs = [], []

            # Fivefold cross validation
            for train_index, test_index in kfold.split(self.X):
                X_train, y_train = self.X.iloc[train_index, :], self.y.iloc[train_index]
                X_test, y_test = self.X.iloc[test_index, :], self.y.iloc[test_index]

                # Resample training data (SMOTER)
                relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=self.bins[0])
                X_train, y_train = resreg.smoter(X_train, y_train, relevance,
                                                 relevance_threshold=0.5, k=k, over=over,
                                                 random_state=0)

                # Fit regressor and evaluate performance
                reg = RandomForestRegressor(n_estimators=50, max_features=0.5, n_jobs=-1,
                                            random_state=0)
                cv_err, train_err = self.implementML(X_train, y_train, X_test, y_test, reg)

                cv_errs.append(cv_err)
                train_errs.append(train_err)
            # Plot results
            title = 'SMOTER {0} {1}'.format(over, k)
            self.plotPerformanceMAE(cv_errs, train_errs, title=title, k=5)
            cv_errs_store.append(np.mean(cv_errs))
            train_errs_store.append(np.mean(train_errs))

        # Determine the best parameters
        best = np.argsort(cv_errs_store)[0]  # Which is the best
        print('''Best parameters:
            over={0}; k={1}'''.format(params[best][0], params[best][1]))
        cv_err, train_err = cv_errs_store[best], train_errs_store[best]

        # Save results
        self.cache['SMOTER'] = [params[best], cv_err]

    def gauss(self):

        # Parameters
        overs = ['balance', 'average', 'extreme']
        deltas = [0.01, 0.1, 0.5]  # amount of Gaussian noise
        params = list(itertools.product(overs, deltas))
        # Empty lists for storing results
        cv_err_store, train_err_store = [], []
        # Grid search
        for over, delta in params:
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            cv_errs, train_errs = [], []

            # Fivefold cross validation
            for train_index, test_index in kfold.split(self.X):
                X_train, y_train = self.X.iloc[train_index, :], self.y.iloc[train_index]
                X_test, y_test = self.X.iloc[test_index, :], self.y.iloc[test_index]

                # Resample training data (Gaussian Noise)
                relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=self.bins[0])
                X_train, y_train = resreg.gaussian_noise(X_train, y_train, relevance,
                                                         relevance_threshold=0.5, delta=delta,
                                                         over=over,
                                                         random_state=0)

                # Fit regressor and evaluate performance
                reg = RandomForestRegressor(n_estimators=50, max_features=0.5, n_jobs=-1,
                                            random_state=0)
                cv_err, train_err = self.implementML(X_train, y_train, X_test, y_test, reg)

                cv_errs.append(cv_err)
                train_errs.append(train_err)
            title = 'Gaussian noise (GN) {0} {1}'.format(over, delta)
            self.plotPerformanceMAE(cv_errs, train_errs, title=title, k=5)

            cv_err, train_err = np.mean(cv_errs), np.mean(train_errs)

            # Store grid search results
            cv_err_store.append(cv_err)
            train_err_store.append(train_err)


        # Determine the best parameters
        best = np.argsort(cv_err_store)[0]  # Which is the best
        print('''Best parameters:
            over={0}; delta={1}'''.format(params[best][0], params[best][1]))
        cv_err, train_err = cv_err_store[best], train_err_store[best]

        # Save results
        self.cache['GN'] = [params[best], cv_err]



    def wercs(self):
        # Parameters
        overs = [0.5, 0.75, 1.0]  # percent of samples added
        unders = [0.5, 0.75]  # percent of samples removed
        noises = [True, False]  # Whether to add Gaussian noise to oversampled data
        deltas = [0.01, 0.1, 0.5]  # amount of Gaussian noise
        params = list(itertools.product(overs, unders, [noises[1]])) + \
                 list(itertools.product(overs, unders, [noises[0]], deltas))

        # Empty lists for storing results
        cv_err_store, train_err_store = [], []

        # Grid search
        for param in params:
            if len(param) == 4:
                over, under, noise, delta = param
            else:
                over, under, noise = param
                delta = None
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            cv_errs, train_errs = [], []

            # Fivefold cross validation
            for train_index, test_index in kfold.split(self.X):
                X_train, y_train = self.X.iloc[train_index, :], self.y.iloc[train_index]
                X_test, y_test = self.X.iloc[test_index, :], self.y.iloc[test_index]

                # Resample training data (WERCS)
                relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=self.bins[0])
                X_train, y_train = resreg.wercs(X_train, y_train, relevance, over=over,
                                                under=under, noise=noise, delta=delta,
                                                random_state=0)

                # Fit regressor and evaluate performance
                reg = RandomForestRegressor(n_estimators=50, max_features=0.5, n_jobs=-1,
                                            random_state=0)
                cv_err, train_err = self.implementML(X_train, y_train, X_test, y_test, reg)
                cv_errs.append(cv_err)
                train_errs.append(train_err)
            cv_err, train_err = np.mean(cv_errs), np.mean(train_errs)
            # Plot results
            self.plotPerformanceMAE(cv_errs, train_errs, title='WERCS', k=5)

            # Store grid search results
            cv_err_store.append(cv_err)
            train_err_store.append(train_err)

        # Determine the best parameters
        best = np.argsort(cv_err_store)[0]  # Which is the best
        bestparam = params[best]
        if len(bestparam) == 4:
            over, under, noise, delta = bestparam
        else:
            over, under, noise = bestparam
            delta = None
        print(f'''Best parameters:
            over={over}; under={under}; noise={noise}; delta={delta}''')
        cv_err, train_err = cv_err_store[best], train_err_store[best]


        # Save results
        self.cache['WERCS'] = [params[best], cv_err]

    def plot_overall(self):
        # Data from CACHE
        best = [val[0] for val in self.cache.values()]
        cv_errs = [val[1] for val in self.cache.values()]

        keys = self.cache.keys()

        # # Plot r2

