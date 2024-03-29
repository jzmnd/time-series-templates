"""Time series data class"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.ts_functions import SeriesToSupervisedTransformer, DifferenceTransformer


class TSData:
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

        # Parameters for supervised learning problem
        self.look_back = look_back
        self.look_fwd = look_fwd

        self._standardized_data()

    def _standardized_data(self):
        """Expect first column to be time and second column to be y data and convert the time
        column to a datetime object."""
        self.dt_col, self.y_col = self.data.columns
        self.data[self.dt_col] = pd.to_datetime(self.data[self.dt_col])

    def reset_data(self):
        """Reset data to the original raw data."""
        self.data = self._raw_data.copy()
        self._standardized_data()

    @classmethod
    def from_file(cls, filename, test_split=12, look_back=0, look_fwd=12):
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
        """Perform min-max scaling on the data."""
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
        """Perform 1st order differencing on the data."""
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

    def series_to_supervised(self):
        """Convert series data to supervised learning problem by adding lagged observations as
        input and future observations as targets.
        """
        self.sersup = SeriesToSupervisedTransformer(n_in=self.look_back, n_out=self.look_fwd)

        y_trans = self.sersup.fit_transform(self._y_data_arr)

        # create correct index and column names for the transformed data
        index = range(self.look_back + 1, self.data.shape[0] - self.look_fwd + 1)
        columns = [
            "{:s}(t{:+d})".format(self.y_col, i) for i in range(-self.look_back, self.look_fwd)
        ]

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
        return self.data.iloc[: -self.test_split].copy()

    @property
    def test_X(self):
        """The training features data.

        Returns:
            DataFrame
        """
        return self.data.filter(like=self.y_col).iloc[-self.test_split:, :self.look_back].copy()

    @property
    def test_y(self):
        """The training labels data.

        Returns:
            DataFrame
        """
        return self.data.filter(like=self.y_col).iloc[-self.test_split:, self.look_back:].copy()

    @property
    def train_X(self):
        """The test features data.

        Returns:
            DataFrame
        """
        return self.data.filter(like=self.y_col).iloc[:-self.test_split, :self.look_back].copy()

    @property
    def train_y(self):
        """The test labels data.

        Returns:
            DataFrame
        """
        return self.data.filter(like=self.y_col).iloc[:-self.test_split, self.look_back:].copy()
