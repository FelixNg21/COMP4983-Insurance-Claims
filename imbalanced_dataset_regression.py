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

class ImbalancedDatasetReg:
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
        regressor (class): The regressor class to be used.
        print_results (bool): Whether to print the results or not.

    """

    def __init__(self, path, col, threshold,
                 regressor, performance=False, print_results=False):
        self.data = pd.read_csv(path)
        self.data = self.data.iloc[:, 1:]
        self.X = self.data.drop(col, axis='columns', inplace=False)
        self.y = self.data.loc[:, col]
        self.bins = [threshold]
        self.cache = {}
        self.performance = performance
        self.regressor = regressor
        self.print_results = print_results

    def implementML(self, x_train, y_train, x_test, y_test, func_name):
        """
        Implements machine learning regression using the specified regressor.

        Args:
            self: The instance of the LinearModel class.
            x_train (numpy.ndarray): The training feature data.
            y_train (numpy.ndarray): The training label data.
            x_test (numpy.ndarray): The testing feature data.
            y_test (numpy.ndarray): The testing label data.
            func_name (str): The name of the resampling technique.

        Returns:
            tuple: A tuple containing the cross-validation error and training error.

        """

        self.regressor.fit(x_train, y_train)  # fit regressor
        y_pred = self.regressor.predict(x_test)

        if self.print_results:
            df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            df.to_csv(f'{func_name}_output.csv')

        cv_err = np.mean(np.abs(y_test - y_pred))
        train_err = np.mean(np.abs(y_train - self.regressor.predict(x_train)))
        return cv_err, train_err

    def plotPerformanceMAE(self, cv_err, train_err, title, k=5):
        plt.plot(list(range(k)), cv_err, color='royalblue', linewidth=1, label='CV error')
        plt.plot(list(range(k)), train_err, color='crimson', linewidth=1, label='Train error')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.xlabel('K')
        plt.title(title + " MAE vs Neighbours")
        plt.legend()
        plt.show()
        plt.close()

    def no_resampling(self):
        """
        Performs regression without any resampling technique and evaluates the performance of the regressor.

        Args:
            self: The instance of the LinearModel class.

        Returns:
            None

        Explanation:
            This method performs regression without any resampling technique. It performs five-fold cross-validation and fits the regressor on the training data. The performance of the regressor is evaluated using the implementML method. The mean absolute errors for cross-validation and training are calculated and stored. If the performance flag is set, it plots the performance results. Finally, it determines the best performing fold based on the cross-validation errors and saves the results in the cache.

        """
        # Empty list for storing results
        cv_errs, train_errs = [], []

        # Fivefold cross validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)

        for train_index, test_index in kfold.split(self.X):
            x_train, y_train = self.X.iloc[train_index, :], self.y.iloc[train_index]
            x_test, y_test = self.X.iloc[test_index, :], self.y.iloc[test_index]

            cv_err, train_err = self.implementML(x_train, y_train, x_test, y_test, self.no_resampling.__name__)
            cv_errs.append(cv_err)
            train_errs.append(train_err)

        # Average performance
        min_index = np.argmin(cv_errs)
        min_cv_err = cv_errs[min_index]

        # View performance
        if self.performance:
            self.plotPerformanceMAE(cv_errs, train_errs, k=5, title='No resampling (None)')
        print('No resampling (None) mae: {0}'.format(min_cv_err))
        # Save performance results
        self.cache['None'] = [min_index, min_cv_err]

    def smoter(self):
        """
        Performs grid search using SMOTER resampling and evaluates the performance of the regressor.

        Args:
            self: The instance of the LinearModel class.

        Returns:
            None
        """
        # Parameters
        overs = ['balance'] # orig: 'balance', 'average', 'extreme'
        ks = [1]  # nearest neighbors, determined to be 1
        params = list(itertools.product(overs, ks))
        cv_errs_store, train_errs_store = [], []

        # Grid search
        for over, k in params:
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            cv_errs, train_errs = [], []

            # Fivefold cross validation
            for train_index, test_index in kfold.split(self.X):
                x_train, y_train = self.X.iloc[train_index, :], self.y.iloc[train_index]
                x_test, y_test = self.X.iloc[test_index, :], self.y.iloc[test_index]
                # Resample training data (SMOTER)
                relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=self.bins[0])
                x_train_smoter, y_train_smoter = resreg.smoter(x_train, y_train, relevance,
                                                 relevance_threshold=0.5, k=k, over=over,
                                                 random_state=0)
                # Fit regressor and evaluate performance
                cv_err, train_err = self.implementML(x_train_smoter, y_train_smoter, x_test, y_test, self.smoter.__name__)

                cv_errs.append(cv_err)
                train_errs.append(train_err)
            # Plot results
            if self.performance:
                title = 'SMOTER {0} {1}'.format(over, k)
                self.plotPerformanceMAE(cv_errs, train_errs, title=title, k=5)
            cv_errs_store.append(np.mean(cv_errs))
            train_errs_store.append(np.mean(train_errs))

        # Determine the best parameters
        best = np.argsort(cv_errs_store)[0]  # Which is the best
        print('''Best parameters:
            over={0}; k={1}; mae={2}'''.format(params[best][0], params[best][1], cv_errs_store[best]))
        cv_err, train_err = cv_errs_store[best], train_errs_store[best]

        # Save results
        self.cache['SMOTER'] = [params[best], cv_err]

    def gauss(self):
        """
        Performs grid search using Gaussian noise resampling and evaluates the performance of the regressor.

        Args:
            self: The instance of the LinearModel class.

        Returns:
            None
        """
        # Parameters
        overs = ['balance'] # orig 'balance, 'average', 'extreme'
        deltas = [0.1]  # amount of Gaussian noise #orig: 0.01, 0.1, 0.5
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
                cv_err, train_err = self.implementML(X_train, y_train, X_test, y_test, self.gauss.__name__)

                cv_errs.append(cv_err)
                train_errs.append(train_err)
            if self.performance:
                title = 'Gaussian noise (GN) {0} {1}'.format(over, delta)
                self.plotPerformanceMAE(cv_errs, train_errs, title=title, k=5)

            cv_err, train_err = np.mean(cv_errs), np.mean(train_errs)

            # Store grid search results
            cv_err_store.append(cv_err)
            train_err_store.append(train_err)

        # Determine the best parameters
        best = np.argsort(cv_err_store)[0]  # Which is the best
        print('''Best parameters:
            over={0}; delta={1}; mae={2}'''.format(params[best][0], params[best][1], cv_err_store[best]))
        cv_err, train_err = cv_err_store[best], train_err_store[best]

        # Save results
        self.cache['GN'] = [params[best], cv_err]

    def wercs(self):
        """
        Performs grid search using the WERCS algorithm for resampling and evaluates the performance of the regressor.

        Args:
            self: The instance of the LinearModel class.

        Returns:
            None
        """
        # Parameters
        overs = [0.5]  # percent of samples added, orig: 0.5, 0.75, 1.0
        unders = [0.5]  # percent of samples removed, orig: 0.5, 0.75
        noises = [True]  # Whether to add Gaussian noise to oversampled data, orig: True, False
        deltas = [0.01]  # amount of Gaussian noise, orig: 0.01, 0.1, 0.5
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
                cv_err, train_err = self.implementML(X_train, y_train, X_test, y_test, self.wercs.__name__)
                cv_errs.append(cv_err)
                train_errs.append(train_err)
            cv_err, train_err = np.mean(cv_errs), np.mean(train_errs)
            # Plot results

            if self.performance:
                title = 'WERCS {0} {1} {2} {3}'.format(over, under, noise, delta)
                self.plotPerformanceMAE(cv_errs, train_errs, title=title, k=5)

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
            over={over}; under={under}; noise={noise}; delta={delta}; mae={cv_err_store[best]}''')
        cv_err, train_err = cv_err_store[best], train_err_store[best]

        # Save results
        self.cache['WERCS'] = [params[best], cv_err]

