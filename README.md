# time-series-templates

A notebook of simple templates for time series modeling.
The examples use the air passengers dataset and all perform and test on 12 month ahead forecasts.

`src/ts_data.py` provides a helper class for the time series data.

`src/ts_functions.py` provides scikit-learn transformers for differencing and converting
time series data into supervised learning data for ML models.

`src/minmax_scaling.py` is a simple scaling model based on the minimum and maximum in
each time period.

## List of models

1. Baseline (persistence with seasonality)
1. Min max scaling model
1. Prophet model
1. Seasonal ARIMA
1. Holt-Winters
1. LSTM
1. Windowed MLP
1. Gradient boosted trees
