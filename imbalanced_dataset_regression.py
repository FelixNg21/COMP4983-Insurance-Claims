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

        if over is not None and k is not None:
            df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            df.to_csv(f'{over}_{k}_output.csv')
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        rel_test = resreg.sigmoid_relevance(y_test, cl=None,
                                            ch=self.bins[0])  # relevance values of y_test
        rel_pred = resreg.sigmoid_relevance(y_pred, cl=None,
                                            ch=self.bins[0])  # relevance values of y_pred
        f1 = resreg.f1_score(y_test, y_pred, error_threshold=0.5, relevance_true=rel_test,
                             relevance_pred=rel_pred, relevance_threshold=0.5)
        msebin = resreg.bin_performance(y_test, y_pred, bins=self.bins, metric='mse')
        return r2, f1, msebin, mae

    def plotPerformanceMAE(self, maebin, maebinerr, f1, r2, title):
        plt.bar(range(2), maebin, yerr=maebinerr, width=0.4, capsize=3, color='royalblue',
                linewidth=1, edgecolor='black')
        plt.xlim(-0.5, len(self.bins) + 0.5)
        # plt.xticks(range(2), ['< {0}'.format(self.bins[0]), '≥ {0}'.format(self.bins[0])],
        #            **self.ticks_font)
        plt.yticks(**self.ticks_font)
        plt.ylabel('Mean Average Error (MAE)', self.label_font)
        plt.xlabel('Target value range', self.label_font)
        title = title + '\nf1={0}, r2={1}'.format(round(f1, 4), round(r2, 4))
        plt.title(title, self.title_font)
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
        r2s, f1s, msebins, maebins = [], [], [], []

        # Fivefold cross validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)

        for train_index, test_index in kfold.split(self.X):
            X_train, y_train = self.X.iloc[train_index, :], self.y.iloc[train_index]
            X_test, y_test = self.X.iloc[test_index, :], self.y.iloc[test_index]
            reg = RandomForestRegressor(n_estimators=50, max_features=0.5, n_jobs=-1,
                                        random_state=0)
            r2, f1, msebin, mae = self.implementML(X_train, y_train, X_test, y_test,
                                              reg)  # Fit regressor and evaluate performance
            r2s.append(r2)
            f1s.append(f1)
            msebins.append(msebin)
            maebins.append(mae)

        # Average performance
        r2, f1, msebin, maebin = np.mean(r2s), np.mean(f1s), np.mean(msebins, axis=0), np.mean(maebins)
        # Standard error of the mean
        r2err, f1err, msebinerr, maebinerr = np.std(r2s) / np.sqrt(5), np.std(f1s) / np.sqrt(5), \
                                  np.std(msebins, axis=0) / np.sqrt(5), np.std(maebins) / np.sqrt(5)
        # View performance
        self.plotPerformanceMAE(maebin, maebinerr, f1, r2, title='No resampling (None)')

        # Save performance results
        self.cache['None'] = [r2, f1, msebin, r2err, f1err, msebinerr, maebin, maebinerr]

    def smoter(self):
        # Parameters
        overs = ['balance', 'average', 'extreme']
        ks = [x for x in range(1, 10) if x % 2]  # nearest neighbors
        params = list(itertools.product(overs, ks))

        # Empty lists for storing results
        r2store, f1store, msebinstore, maestore = [], [], [], []
        r2errstore, f1errstore, msebinerrstore, maebinerrstore = [], [], [], []

        # Grid search
        for over, k in params:
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            r2s, f1s, msebins, maebins = [], [], [], []

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
                r2, f1, msebin, mae = self.implementML(X_train, y_train, X_test, y_test, reg, over, k)
                r2s.append(r2)
                f1s.append(f1)
                msebins.append(msebin)
            r2, f1, msebin, maebin = np.mean(r2s), np.mean(f1s), np.mean(msebins, axis=0), np.mean(maebins)
            r2err, f1err, msebinerr, maebinerr = np.std(r2s) / np.sqrt(5), np.std(f1s) / np.sqrt(5), \
                                      np.std(msebins, axis=0) / np.sqrt(5), np.std(maebins) / np.sqrt(5)

            # Store grid search results
            r2store.append(r2)
            f1store.append(f1)
            msebinstore.append(msebin)
            maestore.append(maebin)
            r2errstore.append(r2err)
            f1errstore.append(f1err)
            msebinerrstore.append(msebinerr)
            maebinerrstore.append(maebinerr)

        # Determine the best parameters
        best = np.argsort(maestore)[-1]  # Which is the best
        print('''Best parameters:
            over={0}; k={1}'''.format(params[best][0], params[best][1]))
        f1, r2, msebin, maebin = f1store[best], r2store[best], msebinstore[best], maestore[best]
        f1err, r2err, msebinerr, maebinerr = f1errstore[best], r2errstore[best], msebinerrstore[best], maebinerrstore[best]

        # Save results
        self.cache['SMOTER'] = [r2, f1, msebin, r2err, f1err, msebinerr, maebin, maebinerr]

        # Plot results
        self.plotPerformanceMAE(maebin, maebinerr, f1, r2, title='SMOTER')

    def gauss(self):

        # Parameters
        overs = ['balance', 'average', 'extreme']
        deltas = [0.01, 0.1, 0.5]  # amount of Gaussian noise
        params = list(itertools.product(overs, deltas))

        # Empty lists for storing results
        r2store, f1store, msebinstore, maebinstore = [], [], [], []
        r2errstore, f1errstore, msebinerrstore, maebinerrstore = [], [], [], []

        # Grid search
        for over, delta in params:
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            r2s, f1s, msebins, maebins = [], [], [], []

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
                r2, f1, msebin, maebin = self.implementML(X_train, y_train, X_test, y_test, reg)
                r2s.append(r2)
                f1s.append(f1)
                msebins.append(msebin)
                maebins.append(maebin)
            r2, f1, msebin, maebin = np.mean(r2s), np.mean(f1s), np.mean(msebins, axis=0), np.mean(maebins)
            r2err, f1err, msebinerr, maebinerr = np.std(r2s) / np.sqrt(5), np.std(f1s) / np.sqrt(5), \
                                      np.std(msebins, axis=0) / np.sqrt(5), np.std(maebins) / np.sqrt(5)

            # Store grid search results
            r2store.append(r2)
            f1store.append(f1)
            msebinstore.append(msebin)
            maebinstore.append(maebin)
            r2errstore.append(r2err)
            f1errstore.append(f1err)
            msebinerrstore.append(msebinerr)
            maebinerrstore.append(maebinerr)

        # Determine the best parameters
        best = np.argsort(maebinstore)[-1]  # Which is the best
        print('''Best parameters:
            over={0}; delta={1}'''.format(params[best][0], params[best][1]))
        f1, r2, msebin, maebin = f1store[best], r2store[best], msebinstore[best], maebinstore[best]
        f1err, r2err, msebinerr, maebinerr = f1errstore[best], r2errstore[best], msebinerrstore[best], maebinerrstore[best]

        # Save results
        self.cache['GN'] = [r2, f1, msebin, r2err, f1err, msebinerr, maebin, maebinerr]

        # Plot results
        self.plotPerformanceMAE(maebin, maebinerr, f1, r2, title='Gaussian noise (GN)')

    def wercs(self):
        # Parameters
        overs = [0.5, 0.75, 1.0]  # percent of samples added
        unders = [0.5, 0.75]  # percent of samples removed
        noises = [True, False]  # Whether to add Gaussian noise to oversampled data
        deltas = [0.01, 0.1, 0.5]  # amount of Gaussian noise
        params = list(itertools.product(overs, unders, [noises[1]])) + \
                 list(itertools.product(overs, unders, [noises[0]], deltas))

        # Empty lists for storing results
        r2store, f1store, msebinstore, maebinstore = [], [], [], []
        r2errstore, f1errstore, msebinerrstore, maebinerrorstore = [], [], [], []

        # Grid search
        for param in params:
            if len(param) == 4:
                over, under, noise, delta = param
            else:
                over, under, noise = param
                delta = None
            kfold = KFold(n_splits=5, shuffle=True, random_state=0)
            r2s, f1s, msebins, maebins = [], [], [], []

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
                r2, f1, msebin, maebin = self.implementML(X_train, y_train, X_test, y_test, reg)
                r2s.append(r2)
                f1s.append(f1)
                msebins.append(msebin)
                maebins.append(maebin)
            r2, f1, msebin, maebin = np.mean(r2s), np.mean(f1s), np.mean(msebins, axis=0), np.mean(maebins)
            r2err, f1err, msebinerr, maebinerr = np.std(r2s) / np.sqrt(5), np.std(f1s) / np.sqrt(5), \
                                      np.std(msebins, axis=0) / np.sqrt(5), np.std(maebins) / np.sqrt(5)

            # Store grid search results
            r2store.append(r2)
            f1store.append(f1)
            msebinstore.append(msebin)
            maebinstore.append(maebin)
            r2errstore.append(r2err)
            f1errstore.append(f1err)
            msebinerrstore.append(msebinerr)
            maebinerrorstore.append(maebinerr)

        # Determine the best parameters
        best = np.argsort(maebinstore)[-1]  # Which is the best
        bestparam = params[best]
        if len(bestparam) == 4:
            over, under, noise, delta = bestparam
        else:
            over, under, noise = bestparam
            delta = None
        print(f'''Best parameters:
            over={over}; under={under}; noise={noise}; delta={delta}''')
        f1, r2, msebin, maebin = f1store[best], r2store[best], msebinstore[best], maebinstore[best]
        f1err, r2err, msebinerr, maebinerr = f1errstore[best], r2errstore[best], msebinerrstore[best], maebinerrorstore[best]

        # Save results
        self.cache['WERCS'] = [r2, f1, msebin, r2err, f1err, msebinerr, maebin, maebinerr]

        # Plot results
        self.plotPerformanceMAE(maebin, maebinerr, f1, r2, title='WERCS')

    def plot_overall(self):
        # Data from CACHE
        r2s = [val[0] for val in self.cache.values()]
        r2errs = [val[3] for val in self.cache.values()]
        f1s = [val[1] for val in self.cache.values()]
        f1errs = [val[4] for val in self.cache.values()]
        msebins = np.asarray([val[2] for val in self.cache.values()])
        msebinerrs = np.asarray([val[5] for val in self.cache.values()])
        maebins = np.asarray([val[6] for val in self.cache.values()])
        maebinerrs = np.asarray([val[7] for val in self.cache.values()])
        keys = self.cache.keys()

        # # Plot r2
        plt.bar(range(len(keys)), r2s, yerr=r2errs, capsize=3, color='royalblue',
                linewidth=1, edgecolor='black')
        _ = plt.xticks(range(len(keys)), keys, rotation=45)
        plt.ylabel('R2')
        plt.show()

        # Plot F1
        plt.bar(range(len(keys)), f1s, yerr=f1errs, capsize=3, color='crimson',
                linewidth=1, edgecolor='black')
        _ = plt.xticks(range(len(keys)), keys, rotation=45)
        plt.ylabel('F1 score')
        plt.show()

        # Plot MSE over bins
        plt.bar(np.arange(len(keys)) - 0.2, msebins[:, 0], width=0.4, yerr=msebinerrs[:, 0],
                capsize=3, color='goldenrod', linewidth=1, edgecolor='black', label='y<3.5')
        plt.bar(np.arange(len(keys)) + 0.2, msebins[:, 1], width=0.4, yerr=msebinerrs[:, 1],
                capsize=3, color='green', linewidth=1, edgecolor='black', label='y>3.5')
        _ = plt.xticks(range(len(keys)), keys, rotation=45)
        plt.ylabel('Mean squared error')

        # Plot MAE
        plt.bar(np.arange(len(keys)) - 0.2, maebins, width=0.4, yerr=maebinerrs,
                capsize=3, color='goldenrod', linewidth=1, edgecolor='black', label='y<3.5')
        _ = plt.xticks(range(len(keys)), keys, rotation=45)
        plt.ylabel('Mean absolute error')
        plt.legend()
        plt.show()
