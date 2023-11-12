import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler


class LinearModel:
    """
    LinearModel class for performing ridge and lasso regression.

    Args:
        data (pandas.DataFrame): The input data for regression.
        ratio (float, optional): The ratio of data to be used for training. Defaults to 0.30.

    Attributes:
        data_processed (pandas.DataFrame): The preprocessed data.
        scalar (StandardScaler): The standard scaler for feature scaling.
        data (pandas.DataFrame): The input data.
        ratio (float): The ratio of data to be used for training.
        train_data (pandas.DataFrame): The training data.
        test_data (pandas.DataFrame): The testing data.
        ridge_alpha (float): The optimal alpha value for ridge regression.
        lasso_alpha (float): The optimal alpha value for lasso regression.

    Methods:
        preprocess: Preprocesses the input data.
        split_data: Splits the preprocessed data into training and testing sets.
        ridge: Performs ridge regression and finds the optimal alpha value.
        lasso: Performs lasso regression and finds the optimal alpha value.
        _perform_reg: Performs ridge or lasso regression with a range of alpha values.
        cv: Performs cross-validation for ridge or lasso regression.

    """
    def __init__(self, data, ratio=0.30):
        self.data_processed = None
        self.scalar = StandardScaler()
        self.data = data
        self.ratio = ratio
        self.preprocess()
        self.train_data, self.test_data = self.split_data()
        self.ridge_alpha = None
        self.lasso_alpha = None

    def preprocess(self):
        """
        Preprocesses the input data.

        Args:
            self: The instance of the LinearModel class.

        Returns:
            None
        """
        feature_data = self.data.iloc[:, :-1]
        label_data = self.data.iloc[:, -1]

        feature_data_std = self.scalar.fit_transform(feature_data)

        merged = pd.concat([pd.DataFrame(feature_data_std), pd.DataFrame(label_data)], axis=1)
        self.data_processed = merged

    def split_data(self):
        """
        Splits the preprocessed data into training and testing sets.

        Args:
            self: The instance of the LinearModel class.

        Returns:
            tuple: A tuple containing the training data and testing data.
        """
        num_rows = self.data_processed.shape[0]
        shuffled_indices = list(range(num_rows))
        np.random.seed(42)
        np.random.shuffle(shuffled_indices)

        train_set_size = int(num_rows * self.ratio)

        train_indices = shuffled_indices[:train_set_size]
        test_indices = shuffled_indices[train_set_size:]

        train_data = self.data_processed.iloc[train_indices, :]
        test_data = self.data_processed.iloc[test_indices, :]

        return train_data, test_data

    def ridge(self, lb, ub):
        self.ridge_alpha = self._perform_reg(lb, ub, Ridge)

    def lasso(self, lb, ub):
        self.lasso_alpha = self._perform_reg(lb, ub, Lasso)

    def _perform_reg(self, lb, ub, regressor):
        """
        Performs regression with a range of alpha values.

        Args:
            self: The instance of the LinearModel class.
            lb (int): The lower bound of the exponent for the lambda values.
            ub (int): The upper bound of the exponent for the lambda values.
            regressor (class): The regression class to be used (Ridge or Lasso).

        Returns:
            float: The optimal alpha value for the regression.
        """
        lambda_values = [10**i for i in range(lb, ub)]
        cv_errors = []
        train_errors = []
        reg = regressor
        for lambda_value in lambda_values:
            cv_error, train_error = self.cv(
                self.train_data.iloc[:, :-1],
                self.train_data.iloc[:, -1],
                5,
                lambda_value,
                reg,
            )
            cv_errors.append(cv_error)
            train_errors.append(train_error)
        lowest_lambda = lambda_values[cv_errors.index(min(cv_errors))]
        print("Lowest lambda value: ", lowest_lambda)
        result = lowest_lambda
        plt.plot(lambda_values, cv_errors, label="cv error")
        plt.plot(lambda_values, train_errors, label="train error")
        plt.xscale("log")
        plt.xlabel("lambda")
        plt.ylabel("error")
        plt.legend()
        plt.show()
        model = reg(alpha=lowest_lambda)
        model.fit(self.train_data.iloc[:, :-1], self.train_data.iloc[:, -1])
        test_pred = model.predict(self.test_data.iloc[:, :-1])
        test_pred_cv_error = np.mean(np.abs(self.test_data.iloc[:, -1] - test_pred))
        print("Test error: ", test_pred_cv_error)
        return result

    def cv(self, x, y, k, alpha, reg):
        """
        Performs cross-validation for ridge or lasso regression.

        Args:
            self: The instance of the LinearModel class.
            x (numpy.ndarray): The feature data.
            y (numpy.ndarray): The label data.
            k (int): The number of folds for cross-validation.
            alpha (float): The alpha value for the regression.
            reg (class): The regression class to be used (Ridge or Lasso).

        Returns:
            tuple: A tuple containing the average cross-validation error and average training error.

        """
        subsets_x = np.array_split(x, k)
        subsets_y = np.array_split(y, k)
        cv_errors = []
        train_errors = []
        reg_model = reg(alpha=alpha)
        for k in range(k):
            train_x = np.concatenate(subsets_x[:k] + subsets_x[k + 1:], axis=0)
            train_y = np.concatenate(subsets_y[:k] + subsets_y[k + 1:], axis=0)
            val_x = subsets_x[k]
            val_y = subsets_y[k]

            reg_model.fit(train_x, train_y)

            train_pred = reg_model.predict(train_x)
            test_pred = reg_model.predict(val_x)

            train_error = np.mean(np.abs(train_y - train_pred))
            test_error = np.mean(np.abs(val_y - test_pred))
            train_errors.append(train_error)
            cv_errors.append(test_error)

        return np.mean(cv_errors), np.mean(train_errors)

    def get_ridge_alpha(self):
        return self.ridge_alpha

    def get_lasso_alpha(self):
        return self.lasso_alpha
