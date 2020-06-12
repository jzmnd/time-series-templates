"""A min-max scaling model for time series data"""
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


class MinMaxScaling():
    """A min-max scaling model for time series data.

    The model is suitable for seasonal data in which there is a clear maximum and minimum during
    the seasonal period and works well if we are particularly interested in fitting those points
    accurately.

    The model first fits a linear regression to all the period maxima and then to all the period
    minima. Next it uses the last `lookback_periods` periods to find a mean time series for the
    season. The prediction use this mean and scales it by the expected minima and maxima that
    are obtained from the two linear regressions.
    """

    def __init__(self, seasonality=12, lookback_periods=1):
        """Initialize model.

        Args:
            seasonality (int, optional): Number of time steps for seasonality
            lookback_periods (int, optional): Number of seasons to lookback for scaling
        """
        self.seasonality = seasonality
        self.lookback_periods = lookback_periods
        self.n_seasons = None
        self.min_model = None
        self.max_model = None
        self.period_model = None

    def _fit_linear(self, ys):
        """Fits a linear regression with constant to `ys`.

        Args:
            ys (np.array)

        Returns:
            RegressionResultsWrapper
        """
        x = add_constant(np.arange(self.n_seasons))
        model = OLS(ys, x).fit()
        return model

    @staticmethod
    def _min_max_norm(y_period):
        """Performs min-max normalization on a series.

        Args:
            y_period (np.array)

        Returns:
            np.array
        """
        y_min = np.min(y_period, axis=1).reshape(-1, 1)
        y_max = np.max(y_period, axis=1).reshape(-1, 1)

        return (y_period - y_min) / (y_max - y_min)

    @staticmethod
    def _min_max_scale(y_period, y_min, y_max):
        """Rescales a normalized series based on a new min and max. Vectorized using outer product
        to handle multiple minima and maxima.

        Args:
            y_period (np.array)
            y_min (np.array)
            y_max (np.array)

        Returns:
            np.array
        """
        return np.outer(y_period, y_max - y_min) + y_min

    def _fit_scaling(self, ys):
        """Fit the scaling model by normalizing each period and taking the mean.

        Args:
            ys (np.array)

        Returns:
            np.array
        """
        ys_norm = self._min_max_norm(ys)
        y_period = ys_norm.mean(axis=0)
        return y_period

    def fit(self, y):
        """Fit method.

        Args:
            y (np.array)
        """
        y_resampled = np.reshape(y.values, (-1, self.seasonality))
        self.n_seasons = len(y_resampled)

        self.min_model = self._fit_linear(y_resampled.min(axis=1))
        self.max_model = self._fit_linear(y_resampled.max(axis=1))

        self.period_model = self._fit_scaling(y_resampled[-self.lookback_periods:])

    def forecast(self, n):
        """Forecast method.

        Args:
            n (int): Number of periods to forecast

        Returns:
            np.array
        """
        x = add_constant(np.arange(self.n_seasons + n))

        min_forecast = self.min_model.predict(x)[-n:]
        max_forecast = self.max_model.predict(x)[-n:]

        scaled_predictions = self._min_max_scale(self.period_model, min_forecast, max_forecast)

        return scaled_predictions.T.flatten()
