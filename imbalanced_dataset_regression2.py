import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import itertools
import resreg  # Assuming you have the resreg library installed

class ImbalancedDatasetReg:
    def __init__(self, path, col, threshold, regressor, param_grid, performance=False, print_results=False):
        self.data = pd.read_csv(path)
        self.data = self.data.iloc[:, 1:]
        self.X = self.data.drop(col, axis='columns', inplace=False)
        self.y = self.data.loc[:, col]
        self.bins = [threshold]
        self.cache = {}
        self.performance = performance
        self.regressor = regressor
        self.param_grid = param_grid
        self.print_results = print_results

    def implementML(self, x_train, y_train, x_test, y_test, func_name):
        self.regressor.fit(x_train, y_train)
        y_pred = self.regressor.predict(x_test)

        if self.print_results:
            df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            df.to_csv(f'{func_name}_output.csv')

        cv_err = mean_absolute_error(y_test, y_pred)
        train_err = mean_absolute_error(y_train, self.regressor.predict(x_train))
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

    def perform_grid_search(self, x_train, y_train, func_name):
        grid_search = GridSearchCV(self.regressor, self.param_grid, cv=5, scoring='neg_mean_absolute_error')
        grid_search.fit(x_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        return best_model

    def resampling_evaluation(self, x_train, y_train, x_test, y_test, func_name):
        best_model = self.perform_grid_search(x_train, y_train, func_name)
        y_pred = best_model.predict(x_test)

        if self.print_results:
            df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            df.to_csv(f'{func_name}_output.csv')

        cv_err = mean_absolute_error(y_test, y_pred)
        train_err = mean_absolute_error(y_train, best_model.predict(x_train))
        return cv_err, train_err

    def no_resampling(self):
        cv_errs, train_errs = [], []
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)

        for train_index, test_index in kfold.split(self.X):
            x_train, y_train = self.X.iloc[train_index, :], self.y.iloc[train_index]
            x_test, y_test = self.X.iloc[test_index, :], self.y.iloc[test_index]

            cv_err, train_err = self.resampling_evaluation(x_train, y_train, x_test, y_test, self.no_resampling.__name__)
            cv_errs.append(cv_err)
            train_errs.append(train_err)

        min_index = np.argmin(cv_errs)
        min_cv_err = cv_errs[min_index]

        if self.performance:
            self.plotPerformanceMAE(cv_errs, train_errs, k=5, title='No resampling (None)')
        print('No resampling (None) MAE: {0}'.format(min_cv_err))
        self.cache['None'] = [min_index, min_cv_err]

    def smoter(self):

        overs = ['balance', 'average', 'extreme']  # orig: 'balance', 'average', 'extreme'
        ks = [1]  # nearest neighbors, determined to be 1
        params = list(itertools.product(overs, ks))
        cv_errs_store, train_errs_store = []

        for over, k in params:
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            cv_errs, train_errs = [], []

            for train_index, test_index in kfold.split(self.X):
                x_train, y_train = self.X.iloc[train_index, :], self.y.iloc[train_index]
                x_test, y_test = self.X.iloc[test_index, :], self.y.iloc[test_index]

                relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=self.bins[0])
                x_train_smoter, y_train_smoter = resreg.smoter(x_train, y_train, relevance,
                                                              relevance_threshold=0.5, k=k, over=over,
                                                              random_state=0)

                cv_err, train_err = self.resampling_evaluation(x_train_smoter, y_train_smoter, x_test, y_test,
                                                               self.smoter.__name__)

                cv_errs.append(cv_err)
                train_errs.append(train_err)

            if self.performance:
                title = 'SMOTER {0} {1}'.format(over, k)
                self.plotPerformanceMAE(cv_errs, train_errs, title=title, k=5)

            cv_errs_store.append(np.mean(cv_errs))
            train_errs_store.append(np.mean(train_errs))

        best = np.argsort(cv_errs_store)[0]
        print('''SMOTE Best parameters:
            over={0}; k={1}; mae={2}'''.format(params[best][0], params[best][1], cv_errs_store[best]))
        cv_err, train_err = cv_errs_store[best], train_errs_store[best]
        self.cache['SMOTER'] = [params[best], cv_err]

    # def smoter(self):
    #     overs = ['balance', 'average', 'extreme']
    #     ks = [1]
    #     param_grid = {'over': overs, 'k': ks}
    #
    #     grid_search = GridSearchCV(estimator=self.regressor, param_grid=param_grid, scoring='neg_mean_absolute_error',
    #                                cv=5)
    #     grid_search.fit(self.X, self.y)
    #
    #     best_over = grid_search.best_params_['over']
    #     best_k = grid_search.best_params_['k']
    #     best_model = grid_search.best_estimator_
    #
    #     cv_errs, train_errs = [], []
    #     kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    #
    #     for train_index, test_index in kfold.split(self.X):
    #         x_train, y_train = self.X.iloc[train_index, :], self.y.iloc[train_index]
    #         x_test, y_test = self.X.iloc[test_index, :], self.y.iloc[test_index]
    #
    #         relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=self.bins[0])
    #         x_train_smoter, y_train_smoter = resreg.smoter(x_train, y_train, relevance,
    #                                                        relevance_threshold=0.5, k=best_k, over=best_over,
    #                                                        random_state=0)
    #
    #         cv_err, train_err = self.resampling_evaluation(x_train_smoter, y_train_smoter, x_test, y_test,
    #                                                        self.smoter.__name__)
    #
    #         cv_errs.append(cv_err)
    #         train_errs.append(train_err)
    #
    #     if self.performance:
    #         title = 'SMOTER {0} {1}'.format(best_over, best_k)
    #         self.plotPerformanceMAE(cv_errs, train_errs, title=title, k=5)
    #
    #     best_cv_err = np.mean(cv_errs)
    #     print('''SMOTE Best parameters:
    #         over={0}; k={1}; mae={2}'''.format(best_over, best_k, best_cv_err))
    #     self.cache['SMOTER'] = [{'over': best_over, 'k': best_k}, best_cv_err]

    def gauss(self):
        overs = ['balance', 'average', 'extreme']
        deltas = [0.01, 0.1, 0.5]
        params = list(itertools.product(overs, deltas))
        cv_err_store, train_err_store = []

        for over, delta in params:
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            cv_errs, train_errs = [], []

            for train_index, test_index in kfold.split(self.X):
                X_train, y_train = self.X.iloc[train_index, :], self.y.iloc[train_index]
                X_test, y_test = self.X.iloc[test_index, :], self.y.iloc[test_index]

                relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=self.bins[0])
                X_train, y_train = resreg.gaussian_noise(X_train, y_train, relevance, relevance_threshold=0.5,
                                                         delta=delta, over=over, random_state=0)

                cv_err, train_err = self.resampling_evaluation(X_train, y_train, X_test, y_test, self.gauss.__name__)

                cv_errs.append(cv_err)
                train_errs.append(train_err)

            if self.performance:
                title = 'Gaussian noise (GN) {0} {1}'.format(over, delta)
                self.plotPerformanceMAE(cv_errs, train_errs, title=title, k=5)

            cv_err, train_err = np.mean(cv_errs), np.mean(train_errs)

            cv_err_store.append(cv_err)
            train_err_store.append(train_err)

        best = np.argsort(cv_err_store)[0]
        print('''Gauss Best parameters:
            over={0}; delta={1}; mae={2}'''.format(params[best][0], params[best][1], cv_err_store[best]))
        cv_err, train_err = cv_err_store[best], train_err_store[best]
        self.cache['GN'] = [params[best], cv_err]

    def wercs(self):
        overs = [0.5, 0.75, 1.0]
        unders = [0.5, 0.75]
        noises = [True, False]
        deltas = [0.01, 0.1, 0.5]
        params = list(itertools.product(overs, unders, [noises[1]])) + \
                 list(itertools.product(overs, unders, [noises[0]], deltas))

        cv_err_store, train_err_store = []

        for param in params:
            if len(param) == 4:
                over, under, noise, delta = param
            else:
                over, under, noise = param
                delta = None
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            cv_errs, train_errs = [], []

            for train_index, test_index in kfold.split(self.X):
                X_train, y_train = self.X.iloc[train_index, :], self.y.iloc[train_index]
                X_test, y_test = self.X.iloc[test_index, :], self.y.iloc[test_index]

                relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=self.bins[0])
                X_train, y_train = resreg.wercs(X_train, y_train, relevance, over=over,
                                                under=under, noise=noise, delta=delta,
                                                random_state=0)

                cv_err, train_err = self.resampling_evaluation(X_train, y_train, X_test, y_test, self.wercs.__name__)
                cv_errs.append(cv_err)
                train_errs.append(train_err)

            cv_err, train_err = np.mean(cv_errs), np.mean(train_errs)

            if self.performance:
                title = 'WERCS {0} {1} {2} {3}'.format(over, under, noise, delta)
                self.plotPerformanceMAE(cv_errs, train_errs, title=title, k=5)

            cv_err_store.append(cv_err)
            train_err_store.append(train_err)

        best = np.argsort(cv_err_store)[0]
        bestparam = params[best]
        if len(bestparam) == 4:
            over, under, noise, delta = bestparam
        else:
            over, under, noise = bestparam
            delta = None
        print(f'''WERCS Best parameters:
            over={over}; under={under}; noise={noise}; delta={delta}; mae={cv_err_store[best]}''')
        cv_err, train_err = cv_err_store[best], train_err_store[best]
        self.cache['WERCS'] = [params[best], cv_err]

class Resampler:
    def __init__(self, data, bin):
        self.data = data
        self.bin = bin
        self.smoter_overs = 'balance'
        self.smoter_k = 1
        self.gauss_overs = 'balance'
        self.gauss_deltas = 0.1
        self.wercs_overs = 0.5
        self.wercs_unders = 0.5
        self.wercs_noises = True
        self.wercs_deltas = 0.01

    def smoter(self):
        x_train = self.data.iloc[:, :-1]
        y_train = self.data.iloc[:, -1]
        relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=self.bin)
        x_train_smoter, y_train_smoter = resreg.smoter(x_train, y_train, relevance, relevance_threshold=0.5,
                                                       k=self.smoter_k, over=self.smoter_overs, random_state=0)
        return x_train_smoter, y_train_smoter

    def gauss(self):
        x_train = self.data.iloc[:, :-1]
        y_train = self.data.iloc[:, -1]
        relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=self.bin)
        x_train, y_train = resreg.gaussian_noise(x_train, y_train, relevance, relevance_threshold=0.5,
                                                 delta=self.gauss_deltas, over=self.gauss_overs, random_state=0)
        return x_train, y_train

    def wercs(self):
        x_train= self.data.iloc[:, :-1]
        y_train = self.data.iloc[:, -1]
        relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=self.bin)
        x_train, y_train = resreg.wercs(x_train, y_train, relevance, over=self.wercs_overs, under=self.wercs_unders,
                                        noise=self.wercs_noises, delta=self.wercs_deltas, random_state=0)
        return x_train, y_train
