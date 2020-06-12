"""Functions for time series templates notebook"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
    """Scikit-learn Transformer for reframing time series data as a supervised learning dataset.
    For more info on the inspiration for this transformer see:
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/"""

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


class TSData():
    """Time series data class for conveniently loading data, applying transformations and
    splitting into training and test sets"""

    def __init__(self, data, test_split, look_back, look_fwd):
        """Initialize data class.

        Args:
            data (DataFrame): Time series data in 2 columns
            test_split (int): Number of points to split as test set from end of data
            look_back (int): Look back period for supervised learning dataset
            look_fwd (int): Forward forecast for supervised learning dataset
        """
        self.data = data
        self._raw_data = self.data.copy()
        self.test_split = test_split
        self.scaler = None
        self.differencer = None
        self.sersup = None

        # first column is time second column is y
        self.dt_col, self.y_col = self.data.columns

        self.data[self.dt_col] = pd.to_datetime(self.data[self.dt_col])

    def reset_data(self):
        """Reset data to the original raw data.
        """
        self.data = self._raw_data.copy()
        self.dt_col, self.y_col = self.data.columns

    @classmethod
    def from_file(cls, filename, test_split=12, look_back=1, look_fwd=12):
        """Load data from a csv file.

        Args:
            filename (str)
            test_split (int, optional)
            look_back (int, optional)
            look_fwd (int, optional)

        Returns:
            TSData
        """
        df = pd.read_csv(filename)
        df.columns = df.columns.str.lower().str.strip("#")
        return cls(df, test_split, look_back, look_fwd)

    @property
    def _y_data_arr(self):
        """Return the y data as an array reshaped to a single column.

        Returns:
            array
        """
        return self.data[self.y_col].values.reshape(-1, 1)

    def min_max_scale(self):
        """Perform min-max scaling on the data.
        """
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data[self.y_col] = self.scaler.fit_transform(self._y_data_arr)

    def inv_min_max_scale(self, y):
        """Inverse transformation of scaler.

        Args:
            y (array): Data to inverse scale

        Returns:
            array
        """
        return self.scaler.inverse_transform(y)

    def difference(self):
        """Perform 1st order differencing on the data.
        """
        self.differencer = DifferenceTransformer()
        self.data[self.y_col] = self.differencer.fit_transform(self._y_data_arr)

    def inv_difference(self, y, y0):
        """Inverse the differencing transformation

        Args:
            y (array): Data to inverse difference

        Returns:
            array
        """
        return self.differencer.inverse_transform(y, y0)

    def series_to_supervised(self, n_in, n_out):
        """Convert series data to supervised learning problem by adding lagged observations as
        input and future observations as targets.

        Args:
            n_in (int): Number of lag observations as input
            n_out (int): Number of future observations as output
        """
        self.sersup = SeriesToSupervisedTransformer(n_in=n_in, n_out=n_out)

        y_trans = self.sersup.fit_transform(self._y_data_arr)

        # create correct index and column names for the transformed data
        index = range(n_in + 1, self.data.shape[0] - n_out + 1)
        columns = ["{:s}(t{:+d})".format(self.y_col, i) for i in range(-n_in, n_out)]

        y_trans = pd.DataFrame(y_trans, index=index, columns=columns)

        self.data = pd.concat([self.data[self.dt_col], y_trans], axis=1).dropna()

    def set_column_names(self, dt_col, y_col):
        """Rename the date and y column names.

        Args:
            dt_col (str): New date column name
            y_col (str): New y column name
        """
        self.data.rename({self.dt_col: dt_col, self.y_col: y_col}, axis=1, inplace=True)
        self.dt_col = dt_col
        self.y_col = y_col

    @property
    def test_data(self):
        """The test data set.

        Returns:
            DataFrame
        """
        return self.data.iloc[-self.test_split:].copy()

    @property
    def train_data(self):
        """The training data set.

        Returns:
            DataFrame
        """
        return self.data.iloc[:-self.test_split].copy()
