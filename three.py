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

    ##########################################################################################################################

    X = data[['K_flag', 'customer_id', 'product_id', 'month', 'units_purchased', 'volume_purchased', 'units_purchased_on_promo']]
    Y = data['net_spend']

    st.write('')
    st.write('')
    # Normalize Dataset
    scaler = MinMaxScaler()

    for col in tqdm(X.columns[5:]):
        X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 123)
    st.write('X_train: {}\nX_test : {}\nY_train: {}\nY_test : {}'.format(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 123)
    st.write('X_train: {}\nX_val  : {}\nY_train: {}\nY_val  : {}'.format(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape))

    # LASSO REGRESSION
    st.write('')
    st.write('')
    st.subheader('LASSO REGRESSION')
    alpha = [10**i for i in range(-5, 2)]

    train_score = []
    val_score = []
    for i in tqdm(alpha):
        
        print("alpha = {} ".format(i))
        _model = Lasso(alpha= i)
        _model.fit(X_train, Y_train)
        rmse_train = mean_squared_error(_model.predict(X_train).clip(0,20), Y_train, squared=False)
        rmse_val = mean_squared_error(_model.predict(X_val).clip(0,20), Y_val, squared=False)

        train_score.append(rmse_train)
        val_score.append(rmse_val)
        print("Training Loss: {} ".format(rmse_train))
        print("Validation Loss: {} ".format(rmse_val))
        print("-"*50)

    params = [str(i) for i in alpha]
    fig, ax = plt.subplots(figsize = (20, 7))
    ax.plot(params, val_score, c='g')
    for i, txt in enumerate(np.round(val_score,3)):
        ax.annotate((params[i],np.round(txt,3)), (params[i],val_score[i]))

    plt.grid()
    plt.title("Cross Validation RMSE for Para Grid")
    plt.xlabel("(Subsample , Cosample_bytree)")
    plt.ylabel("Error Measure")
    st.plotly_charts(ax)

    # RIDGE REGRESSION
    st.write('')
    st.write('')
    st.subheader('RIDGE REGRESSION')
    alpha = [10**i for i in range(-5, 5)]

    train_score = []
    val_score = []
    for i in tqdm(alpha):
        
        print("alpha = {} ".format(i))
        _model = Ridge(alpha= i)
        _model.fit(X_train, Y_train)
        rmse_train = mean_squared_error(_model.predict(X_train).clip(0,20), Y_train, squared=False)
        rmse_val = mean_squared_error(_model.predict(X_val).clip(0,20), Y_val, squared=False)

        train_score.append(rmse_train)
        val_score.append(rmse_val)
        print("Training Loss: {} ".format(rmse_train))
        print("Validation Loss: {} ".format(rmse_val))
        print("-"*50)

    params = [str(i) for i in alpha]
    fig, ax = plt.subplots(figsize = (20, 7))
    ax.plot(params, val_score, c='g')
    for i, txt in enumerate(np.round(val_score,3)):
        ax.annotate((params[i],np.round(txt,3)), (params[i],val_score[i]))

    plt.grid()
    plt.title("Cross Validation RMSE for Para Grid")
    plt.xlabel("(Subsample , Cosample_bytree)")
    plt.ylabel("Error Measure")
    plt.show()