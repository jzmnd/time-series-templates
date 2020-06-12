# time-series-templates

A notebook of simple templates for time series modeling.
The examples use the air passengers dataset and all perform and test on 12 month ahead forecasts.

`ts_functions.py` provides a helper class for the time series data and scikit-learn transformers
for differencing and converting time series data into supervised learning data for neural network
models.

`minmax_scaling.py` is a simple scaling model based on the minimum and maximum in each time period.
