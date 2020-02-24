#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas import DataFrame
import pyodbc
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot
from matplotlib.backends.backend_pdf import PdfPages


# In[3]:


def db_open():
    '''DB Connection details with SQL Server'''
    conn = pyodbc.connect(Driver='{ODBC Driver 17 for SQL Server}',
                          Server='server_name',
                          Port='port_name',
                          Database='database_name',
                          user='username',
                          password="password")
    return conn

# Code for DB connection close.:
def db_close(conn):
    '''Code to close the DB Connection'''
    conn.close()

def read_sql_Table():
    '''Function to get data from Table'''
    conn = db_open()
    try:
        query="select_query"
        df = pd.read_sql(query, conn)  
    except Exception as e:
        print("Exception: "+str(e))
    db_close(conn)
    return(df)


# In[4]:


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    """This method is used to calculate accuracy metrics based on forecasted values """
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE: Mean Absolute Percentage Error
    me = np.mean(forecast - actual)             # ME:Mean Error
   # mae = np.mean(np.abs(forecast - actual))    # MAE:Mean Absolute Error  'mae': [mae],
   # mpe = np.mean((forecast - actual)/actual)   # MPE:Mean Percentage Error 'mpe': [mpe],
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE:Root Mean Squared Error
    corr = np.corrcoef(forecast, actual)[0,1]   # corr:Correlation between the Actual and the Forecast
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax: Min-Max Error
    #acf1 = acf(fc-test)[1]                      # ACF1 'acf1':acf1,
    return({'Mean Absolute Percentage Error':[mape], 'Mean Error':[me], 
             'Root Mean Squared Error':[rmse], 
            'Correlation between Actual-Forecast':[corr], 'Min-Max Error':[minmax]})


# In[5]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    '''Function to test the stationarity of the series'''
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    result = adfuller(timeseries)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


# In[76]:


def plotarima(Train, Test, automodel,pdf_filename,fc,conf):
    '''Function to plot for forecasted and predicted values'''
    linewidthPlot = 2
    linestylePlot = '-'
    markerPlot = 'o'
    colorPlot = 'b'
    
    # Make as pandas series
    fc_series = pd.Series(fc, index=Test.index)
    lower_series = pd.Series(conf[:, 0], index=Test.index)
    upper_series = pd.Series(conf[:, 1], index=Test.index)

    # Plot
    plt.figure(figsize=(10,10), dpi=100)
    plt.plot(Train, label='training',linewidth=linewidthPlot,
              linestyle=linestylePlot, marker=markerPlot)
    plt.plot(Test, label='actual',linewidth=linewidthPlot,
              linestyle=linestylePlot, marker=markerPlot)
    plt.plot(fc_series, label='forecast',linewidth=linewidthPlot,
              linestyle=linestylePlot, marker=markerPlot)
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                     color='k', alpha=.15)
    plt.title("Forecast vs Actuals")
    plt.legend(loc='upper left', fontsize=8)
    return(plt)
    # Save figure to PDF document
    pdf_pages = PdfPages(pdf_filename)
    pdf_pages.savefig(fig)
    pdf_pages.close()
    plt.show()


# # Timeseries

# # Facebook Prophet

# In[8]:


# Get data from sql table for forecast
ts=read_sql_Table()

# select only time and predictor from the imported data in this example, date is 'Time' and 
#predictor(value to be forecasted) is 'Price'.

ts_df=ts[['Time','Price']]


# In[115]:



ts_df_key=ts_df.rename(columns = {'Time':'ds', 'Price':'y'})# rename the columns as required for Prophet
ts_df_key['ds'] = pd.to_datetime(ts_df_key['ds'])  # often date is in string/object format convert in datetime format


split=len(ts_df_key)*0.80      #split data in test and train 20% and 80% respectively.
split=int(split)
Train=ts_df_key[:split]        #in forecast we use lastest data for testing purpose.
Test=ts_df_key[split:]


# In[18]:


#import Prophet

from fbprophet import Prophet 
from fbprophet.plot import plot_plotly

# instantiating a new Prophet object and 
#Prophet is an additive model with the following components:
#y(t) = g(t) + s(t) + h(t) + ϵₜ
# Based on understanding 
#Can add sesonality,holiday and seasonality_mode='multiplicative',.add_seasonality(name='mymonthly',period=30.5,fourier_order=5)

prophet_basic = Prophet(yearly_seasonality = False,weekly_seasonality=True,daily_seasonality=True) 
prophet_basic.add_country_holidays(country_name = 'US')
prophet_basic.fit(Train) #train the algorithm based on training data
future= prophet_basic.make_future_dataframe(periods=len(Test)*2) #predict for future 20 predictions.
forecast=prophet_basic.predict(future)
#forecast=forecast[['yhat','yhat_lower','yhat_upper']] # select desired columns from forecast dataframe

#Merge forecasted values with Test data
merge_forecasted = pd.merge(forecast[['ds','yhat','yhat_lower','yhat_upper']],Test,how='outer',on='ds')

#get the Absolute error % for each Test records.
merge_forecasted['Error%']=np.round(abs(merge_forecasted['y']-merge_forecasted['yhat'])/merge_forecasted['y']*100,2)

#Error the forecasted values in CSV
merge_forecasted.to_csv('forecasted_Prophet.csv', index = False, header=True)

#Visulaze the forecast
fig1 =prophet_basic.plot(forecast)

fig2 = prophet_basic.plot_components(forecast)


# # ARIMA Model

# In[122]:


#check the plot of timesries
ts_df_key['y'].plot()
split=len(ts_df_key)*0.80      #split data in test and train 20% and 80% respectively.
split=int(split)
Train=ts_df_key[:split]        #in forecast we use lastest data for testing purpose.
Test=ts_df_key[split:]


# In[123]:


#Check for Timeseries being stationary
from statsmodels.tsa.stattools import adfuller
print("p-value:", adfuller(ts_df_key['y'].dropna())[1])


# If the p-value is greater than the significance level (0.05),it is not stationary and differencing is as such needed, 
# ie. d > 0.

# In[124]:


#Identify Differencing required(d=?).
from pmdarima.arima.utils import ndiffs

# Estimate the number of differences using an ADF test:
n_adf = ndiffs(ts_df_key['y'], test='adf')  # -> 0

# Or a KPSS test (auto_arima default):
n_kpss = ndiffs(ts_df_key['y'], test='kpss')  # -> 0

print(n_adf)
print(n_kpss) # use the suggessted differencing while training ARIMA Model.


# In[125]:


#verify after "n_adf" timeseries is stationary of not. If p-value is >0.05,timeseries is not stationary.
test_stationarity(ts_df_key['y'].diff(n_adf).dropna(inplace=False))


# The timeseries is stationary at d = 1 where only the first lag is above the significance level.we go on to find out the order of AR, p

# In[126]:


#Autocorrelation Graph of Original timeseries, 1st order differencing timeseries, 2nd order differencing timeseries
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(311)
fig = plot_acf(ts_df_key['y'], ax=ax1,
               title="Autocorrelation on Original Series") 
ax2 = fig.add_subplot(312)
fig = plot_acf(ts_df_key['y'].diff().dropna(), ax=ax2, 
               title="1st Order Differencing")
ax3 = fig.add_subplot(313)
fig = plot_acf(ts_df_key['y'].diff().diff().dropna(), ax=ax3, 
               title="2nd Order Differencing")


# In[127]:


#Get the PACF(for AR(p)) and ACF(for MA(q)) using the stationary timeseries. # Note ACF and PACF is to be observed for stationary timeseries
#When pacf cuts of and acf decays slowly then "AR signature" lag PACF value=p
#When acf cuts of and pacf decays slowly then "MA signature" lag on ACF value=q

plot_pacf(ts_df_key['y'].diff(n_adf).dropna(), lags=30)
plot_acf(ts_df_key['y'].diff(n_adf).dropna(),lags=30)


# In[128]:


forecated_accuracy=pd.DataFrame()
Predicted=pd.DataFrame()
    
model = ARIMA(Train['y'], order=(0,1,1))  #Build ARIMA MODEL
model_fit = model.fit(disp=False)
forecast_arima = model_fit.forecast(steps=len(Test)+10)[0]
Predicted['preds'] =np.round(forecast_arima,2)
Predicted=Predicted.reset_index()
print(model_fit.summary())
# Get the Model Parameters/ Orders used by Auto- Arima
s = model_fit.summary().tables[0].as_text()  
start = s.find("Model:")
end = s.find(")")
end += len("end") 
Predicted['Parameters']=s[start:end] #Store the Model Parameters/ Orders in 'Parameters' column
fc, se, conf = model_fit.forecast(len(Test), alpha=0.05)  # 95% conf
plt=plotarima(Train['y'],Test['y'],model_fit,"ARIMA.pdf",fc,conf)

forecated_accuracy=pd.DataFrame(forecast_accuracy(forecast_arima[:len(Test)], Test['y']))
        
#merge Test,forecasted and accuracy metrics
Test=Test.reset_index(drop = True)
merge_forecasted_arima = pd.merge(Test,Predicted[['preds','Parameters']],how = 'right',left_index = True,right_index = True)
merge_forecasted_arima = pd.merge(merge_forecasted_arima,forecated_accuracy,how = 'left',left_index = True, right_index = True)
merge_forecasted_arima['Error%']=np.round(abs(merge_forecasted_arima['y']-merge_forecasted_arima['preds'])/merge_forecasted_arima['y']*100,2)

merge_forecasted_arima.to_csv('forecasted_arima.csv', index = False, header=True)


# Residual Plot
# 
# We would expect the plot to be random around the value of 0 and not show any trend or cyclic structure.
# we are interested in the mean value of the residual errors. A value close to zero suggests no bias in the forecasts, whereas positive and negative values suggest a positive or negative bias in the forecasts made.

# In[58]:


# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())
# histogram plot
residuals.hist()
plt.show()


# Residual Statistics shows a mean error value close to zero(0.06), but perhaps not close enough.

# In[59]:


autocorrelation_plot(residuals)
plt.show()


# # AUTO ARIMAX

# In[131]:


split=len(ts_df_key)*0.80      #split data in test and train 20% and 80% respectively.
split=int(split)
Train=ts_df_key[:split]        #in forecast we use lastest data for testing purpose.
Test=ts_df_key[split:]


# In[129]:


import pmdarima as pm
def arimamodel(timeseries):
    automodel = pm.auto_arima(timeseries, 
                              start_p=0, 
                              start_q=0,
                              max_p=3,
                              max_q=3,
                              d=None,           # let model determine 'd'
                              test="adf",
                              seasonal=False,
                              trace=True)
    print(automodel.summary())
    return automodel


# In[132]:


forecated_accuracy=pd.DataFrame()
Predicted=pd.DataFrame()
    
model = arimamodel(Train['y'])  #Build Auto ARIMA MODEL
forecast_arima, conf_int = model.predict(n_periods=Test.shape[0]+10, return_conf_int=True)
Predicted['preds'] =np.round(forecast_arima,2)
Predicted=Predicted.reset_index()

# Get the Model Parameters/ Orders used by Auto- Arima
s = model.summary().tables[0].as_text()  
start = s.find("Model:")
end = s.find(")")
end += len("end") 
Predicted['Parameters']=s[start:end] #Store the Model Parameters/ Orders in 'Parameters' column
#forecast
fc, conf = model.predict(len(Test), return_conf_int=True)
plt=plotarima(Train['y'],Test['y'],model,"Auto_ARIMA.pdf",fc, conf)

#Test=Test.reset_index()
forecated_accuracy=pd.DataFrame(forecast_accuracy(forecast_arima[:len(Test)], Test['y']))
        
#merge Test,forecasted and accuracy metrics
Test=Test.reset_index(drop = True)
merge_forecasted_arima = pd.merge(Test,Predicted[['preds','Parameters']],how = 'right',left_index = True, right_index = True)
merge_forecasted_arima = pd.merge(merge_forecasted_arima,forecated_accuracy,how = 'left',left_index = True, right_index = True)
merge_forecasted_arima['Error%']=np.round(abs(merge_forecasted_arima['y']-merge_forecasted_arima['preds'])/merge_forecasted_arima['y']*100,2)

merge_forecasted_arima.to_csv('forecasted_auto_arima.csv', index = False, header=True)


# # SARIMA (Auto-Arima with Seasonality)

# In[133]:


split=len(ts_df_key)*0.80      #split data in test and train 20% and 80% respectively.
split=int(split)
Train=ts_df_key[:split]        #in forecast we use lastest data for testing purpose.
Test=ts_df_key[split:]


# In[134]:


import pmdarima as pm
def sarimamodel(timeseries):
    automodel = pm.auto_arima(timeseries, 
                              start_p=0, 
                              start_q=0,
                              max_p=3,
                              max_q=3,
                              m=7,# weekly seasonality # based on understanding input the correct seasonality value m=1(yearly) m=4(quaterly) m=12(monthly) m=365(daily)
                              d=1,# let model determine 'd'
                              D=1, #force seasonal differencing
                              test="adf",
                              seasonal=True,
                              trace=True,
                              error_action='ignore',  
                              suppress_warnings=True, 
                              stepwise=True)
    print(automodel.summary())
    return automodel


# In[135]:


forecated_accuracy=pd.DataFrame()
Predicted=pd.DataFrame()
    
model = sarimamodel(Train['y'])  #Build Auto ARIMA MODEL  #Build Auto ARIMA MODEL
forecast_sarima, conf_int = model.predict(n_periods=Test.shape[0]+10, return_conf_int=True)
Predicted['preds'] =np.round(forecast_arima,2)
Predicted=Predicted.reset_index()

# Get the Model Parameters/ Orders used by Auto- Arima
s = model.summary().tables[0].as_text()  
start = s.find("Model:")
end = s.find(")")
end += len("end") 
Predicted['Parameters']=s[start:end] #Store the Model Parameters/ Orders in 'Parameters' column
fc, conf = model.predict(len(Test), return_conf_int=True)
plt=plotarima(Train['y'],Test['y'],model,"Auto_SARIMA.pdf",fc, conf)

Test=Test.reset_index()
forecated_accuracy=pd.DataFrame(forecast_accuracy(forecast_sarima[:len(Test)], Test['y']))
        
#merge Test,forecasted and accuracy metrics
Test=Test.reset_index(drop = True)
merge_forecasted_sarima = pd.merge(Test,Predicted[['preds','Parameters']],how = 'right',left_index = True, right_index = True)
merge_forecasted_sarima = pd.merge(merge_forecasted_sarima,forecated_accuracy,how = 'left',left_index = True, right_index = True)
merge_forecasted_sarima['Error%']=np.round(abs(merge_forecasted_sarima['y']-merge_forecasted_sarima['preds'])/merge_forecasted_sarima['y']*100,2)

merge_forecasted_arima.to_csv('forecasted_auto_sarima.csv', index = False, header=True)


# In[137]:


# plot residual errors
autocorrelation_plot(residuals)
plt.show()


# In[ ]:




