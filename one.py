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
    
    st.write('')
    st.write('')
    # Product Data
    st.header('PRODUCTS DATA')

    path = 'product.csv'
    products = pd.read_csv(path)
    st.dataframe(products)
    st.write("Data Shape: {}\n".format(products.shape))

    st.subheader('Data Description')
    st.dataframe(products.describe())

    st.write('')
    st.write('')
    # Stores Data
    st.header('STORES DATA')
    
    path = 'stores.csv'
    stores = pd.read_csv(path)
    st.dataframe(stores)
    st.write("Data Shape: {}\n".format(stores.shape))

    st.subheader('Data Description')
    st.dataframe(stores.describe())

    st.write('')
    st.write('')
    # Customers Data
    st.header('CUSTOMERS DATA')
    
    path = 'customer_supplement.csv'
    customers = pd.read_csv(path)
    st.dataframe(customers)
    st.write("Data Shape: {}\n".format(customers.shape))

    st.subheader('Data Description')
    st.dataframe(customers.describe())

    st.write('')
    st.write('')
    # Sales Data
    st.header('SALES DATA')
    
    path = 'sales.csv'
    sales = pd.read_csv(path)
    st.dataframe(sales)
    st.write("Data Shape: {}\n".format(sales.shape))

    st.subheader('Data Description')
    st.dataframe(sales.describe())

    st.write('')
    st.write('')
    # Merging Data
    data = sales.merge(customers, 
                   on='customer_id', how='left').merge(products, 
                                                       on='product_id', how='left').merge(stores, 
                                                                                          on='store_id', how='left')
    data['date'] = pd.to_datetime(dict(year=2023, month=data.month, day=1))
    data['revenue'] = data.units_purchased*data.net_spend
    st.write("Data Shape: {}\n".format(sales.shape))

    st.title('ABOUT DATA')
    st.subheader('Data Description')
    st.write("We have Records from Date {} to {}".format(data.date.min().date(), data.date.max().date()))
    st.write('')
    st.dataframe(sales.describe())

    st.write("Number of Duplicates: {}".format(len(data[data.duplicated()])))
    st.write('')
    st.write('')

    ##########################################################################################################################