# Forecasting
Forecasting using Prophet, ARIMA, AUTO-ARIMA and SARIMA


During our choice of algorithm we have to compaire results derived from each algorithm.
This repository implements 4 popularly used model for Forecasting timeseries data.
One can tune parameters of each model further however this repository consist of simple approach to get the preliminary analysis.


Implementation ofr:
-Facebook's Prophet
-ARIMA
-Auto ARIMA
-SARIMA

Input Data: Code uses SQL server to fetch the timesries data. One need to understand the granularity(hourly,daily,weekly..) of data before feeding it to the model. Use necessary transformations to get desired forecast. Daily data is used in this case to get daily forecast(10 days).
