# FOR ORGANIZING DATA
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import sys
import streamlit as st
import time


# FOR PLOTTING
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_profiling as pp

# FOR MODELING
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit , RandomizedSearchCV

from optuna.integration import lightgbm as lgb_tuner
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# FOR FORECASTING
from pmdarima import auto_arima
# Fit a SARIMAX(0, 1, 1) on the training set
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Load specific evaluation tools
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
  
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

print('Libraries imported.')

sys.setrecursionlimit(100000)
#print("Installed Dependencies")

###########################################################################

def app():
    st.title("KELLOGGS ASSESSMENT 2023")
    
    # st.header("PART 1")
    
    st.subheader("Loading Page....")
    
    label = st.empty()
    bar = st.progress(0)
    
    for i in range(100):
        # Update progress bar with iterations
        label.text(f'Loaded {i+1} %')
        bar.progress(i+1)
        time.sleep(0.01)
    
    ".... and now we're done!!!"
    
    # Product Data
    path = 'product.csv'
    products = pd.read_csv(path)
 
    # Stores Data
    path = 'stores.csv'
    stores = pd.read_csv(path)
 
    # Customers Data
    path = 'customer_supplement.csv'
    customers = pd.read_csv(path)

    # Sales Data    
    path = 'sales.csv'
    sales = pd.read_csv(path)

    # Merging Data
    data = sales.merge(customers, 
                   on='customer_id', how='left').merge(products, 
                                                       on='product_id', how='left').merge(stores, 
                                                                                          on='store_id', how='left')
    data['date'] = pd.to_datetime(dict(year=2023, month=data.month, day=1))
    data['revenue'] = data.units_purchased*data.net_spend

    st.subheader('Data')
    st.dataframe(data.tail(10))

    ##########################################################################################################################

    PRODUCTID = []
    RMSE      = []
    MSE       = []
    FORECAST  = []

    for i in tqdm(data.product_id.unique()):
        try:
            ID = data[data.product_id == i][['date', 'net_spend']]
            ID = pd.DataFrame(ID.groupby('date')['net_spend'].first())
            ID = ID.sort_values('date')
            # print("Data Shape: {}".format(ID.shape))
            # ID.head()
            
            # Fit auto_arima function to the dataset
            stepwise_fit = auto_arima(ID.net_spend, start_p = 1, start_q = 1,
                                        max_p = 3, max_q = 3, m = 12,
                                        start_P = 0, seasonal = False,
                                        d = None, D = 1, trace = True,
                                        error_action ='ignore',    # we don't want to know if an order does not work
                                        suppress_warnings = True,  # we don't want convergence warnings
                                        stepwise = True)           # set to stepwise    
            # To print the summary
            # stepwise_fit.summary()

            train = ID[ID.index <= pd.to_datetime("2023-10-01", format='%Y-%m-%d')]
            test  = ID[ID.index >= pd.to_datetime("2023-10-01", format='%Y-%m-%d')]

            model = SARIMAX(train.net_spend, order = (0, 1, 1))
            result = model.fit()
            # result.summary()

            start = len(train)
            end = len(train) + len(test) - 1

            # Predictions for one-year against the test set
            predictions = result.predict(start, end, typ = 'levels').rename("Predictions")
            # plot predictions and actual values
            # predictions.plot(legend = True)
            test.net_spend.plot(legend = True)
            # rmse(test.net_spend, predictions) # Calculate root mean squared error
            # mean_squared_error(test.net_spend, predictions) # Calculate mean squared error

            # Train the model on the full dataset
            model = SARIMAX(ID.net_spend, order = (0, 1, 1))
            result = model.fit()
            # Forecast for the next 3 years
            forecast = result.predict(start = len(ID)-1, end = (len(ID)) + 3 * 1, typ = 'levels').rename('Forecast')
            
            # Plot the forecast values
            # ID.net_spend.plot(figsize = (12, 5), legend = True)
            # forecast.plot(legend = True)
            PRODUCTID.append(i)
            RMSE.append(rmse(test.net_spend, predictions)) # Calculate root mean squared error
            MSE.append(mean_squared_error(test.net_spend, predictions))
            FORECAST.append(forecast)
        except:
            print("ERROR: {}".format(i))


    FinalData = pd.DataFrame({'Product ID': PRODUCTID, 'RMSE': RMSE, 'MSE': MSE,'Forecast':FORECAST})
    FinalData.sort_values('Product ID', inplace = True)
    # st.subheader('Result Data')
    # st.dataframe(FinalData.head())

    forecast = pd.DataFrame({'Product ID': PRODUCTID, 'Forecast':FORECAST})
    forecast.Forecast = forecast.Forecast.apply(lambda x: x.mean())
    forecast.sort_values('Forecast', inplace = True)
    st.subheader('Forecast Data')
    st.dataframe(forecast.head())

    # pp.ProfileReport(forecast)

    fig = px.line(ID, x = ID.index, y = ID.net_spend)
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=700, xaxis_title="Date", yaxis_title="Net Spend", title_text="Net Spend on a Product ID") 
    st.plotly_chart(fig)

    train = ID[ID.index <= pd.to_datetime("2023-10-01", format='%Y-%m-%d')]
    test  = ID[ID.index >= pd.to_datetime("2023-10-01", format='%Y-%m-%d')]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.net_spend, mode='lines+markers', name='Train', line = dict(color='black')))
    fig.add_trace(go.Scatter(x=test.index, y=test.net_spend, mode='lines+markers', name='Test', line = dict(color='red')))
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=700, xaxis_title="Date", yaxis_title="Net Spend", title_text="Net Spend on a Product ID") 
    st.plotly_chart(fig)