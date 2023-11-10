"""Functions and data transformers for time series templates notebook"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def smape(y_pred, y_actual):
    """Symmetric mean absolute percentage error (sMAPE) scaled for 0 - 100%.

    Args:
        y_pred (list): Predicted y values
        y_actual (list): Actual y values

    Returns:
        float
    """
    return 100 * np.sum(np.abs(y_pred - y_actual) / (np.abs(y_pred) + np.abs(y_actual)))


class SeriesToSupervisedTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn Transformer for re-framing time series data as a supervised learning dataset.
    For more info on the inspiration for this transformer see:
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    """

    def __init__(self, n_in=1, n_out=1, dropna=True):
        """Initialize transformer.

        Args:
            n_in (int, optional): Number of lag observations as input
            n_out (int, optional): Number of future observations as output
            dropna (bool, optional): Whether or not to drop rows with NaN values
        """
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dropna = dropna

    def fit(self, X, y=None):
        """Fit transformer.

        Args:
            X (array): Sequence of observations
            y (None, optional)

        Returns:
            SeriesToSupervisedTransformer
        """
        return self

    def transform(self, X, y=None):
        """Transform data.

        Args:
            X (array): Sequence of observations
            y (None, optional)

        Returns:
            array
        """
        _X = pd.DataFrame(X)
        _X_trans = []

        # input sequence (t-n, ... t-1)
        for i in range(self.n_in, 0, -1):
            _X_trans.append(_X.shift(i))

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.n_out):
            _X_trans.append(_X.shift(-i))

        # combine sequences into single array
        X_trans = pd.concat(_X_trans, axis=1)

        if self.dropna:
            X_trans.dropna(inplace=True)

        return X_trans.values


class DifferenceTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn Transformer for performing differencing on time series data"""

    def fit(self, X, y=None):
        """Fit transformer by saving the first row in the time series which is needed to
        recover the inverse.

        Args:
            X (array): Input data
            y (None, optional)

        Returns:
            DifferenceTransformer
        """
        self._X0 = X[0, :].copy()
        return self

    def transform(self, X, y=None):
        """Transform the data using first differencing.

        Args:
            X (array): Input data
            y (None, optional)

        Returns:
            array
        """
        check_is_fitted(self, "_X0")

        X_shift = np.roll(X, 1, axis=0)
        X_trans = (X - X_shift).astype(float)
        X_trans[0, :] = np.nan

        return X_trans

    def inverse_transform(self, X, X0=None):
        """Inverse of the differencing transformation.

        Args:
            X (array): Input data
            X0 (array, optional): Starting value if different from fitted

        Returns:
            array
        """
        X_inv = X.copy()
        if X0:
            X_inv = np.vstack([X0, X_inv])
        else:
            X_inv[0, :] = self._X0
        return X_inv.cumsum(axis=0)
