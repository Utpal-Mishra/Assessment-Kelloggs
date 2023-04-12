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

    ##### Reduce the amount of data
    def reduce_mem_usage(df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in tqdm(df.columns):
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024**2
        print('\n\nMemory Usage After Optimization: {:.2f} MB'.format(end_mem))
        print('Decreased By: {:.1f}%\n'.format(100 * (start_mem - end_mem) / start_mem))

        return None

    ##########################################################################################################################

    st.write('')
    st.write('')
    st.subheader('DATA CORRELATION')
    st.dataframe(data.corr(method = 'pearson'))
    st.write('From the CORRELATION is was observed that following set of features were found to be relatively highly related:')
    st.write('\n1. Units Perchased')
    st.write('\n2. Volume Purchased')
    st.write('\n3. Net Spend')
    st.write('\n4. Units Purchased on Promo')
    st.write('\n5. Income and Revenue')
    st.write('\n6. Age')
    st.write('\n7. Retired Flag')

    ##########################################################################################################################

    st.write('')
    st.write('')
    st.subheader('OUTLIERS')
    fig = px.box(data.age, points="outliers")
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=700, width=500, xaxis_title="Age", yaxis_title="Percentile", title_text="Boxplot to Visualize Outliers in Age Feature") 
    st.plotly_chart(fig)

    data = data[data.age < 80]
    st.write('Considering the Interquantile Range for the feature attribute AGE and the practical feasibility of a customers age, the Age Range was restricted to be >20 and <80')

    ##########################################################################################################################

    st.write('')
    st.write('')
    st.subheader('INCOME VS STORE ID')
    storeGB = pd.DataFrame(data.groupby(['store_id'])['income'].first()).reset_index()
    reduce_mem_usage(storeGB)

    fig = px.bar(storeGB, x = storeGB.store_id, y = storeGB.income, color = storeGB.income)
    fig.update_xaxes(rangeslider_visible=True, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=600, width=1400, xaxis_title="Store ID", yaxis_title="Income", title_text="Boxplot to Visualize Customer Income wr.rt. Store ID") 
    st.plotly_chart(fig)

    ##########################################################################################################################
    
    st.write('')
    st.write('')
    st.subheader('INCOME VS AGE')
    ageGB = pd.DataFrame(data.groupby(['age'])['income'].first()).reset_index()
    reduce_mem_usage(ageGB)

    fig = px.bar(ageGB, x = ageGB.age, y = ageGB.income, color = ageGB.income)
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=1400, xaxis_title="Age", yaxis_title="Income", title_text="Barplot to Visualize Customer Income over Different Ages") 
    st.plotly_chart(fig)

    ##########################################################################################################################

    st.write('')
    st.write('')
    st.subheader('INCOME VS PRODUCT ID')
    productGB = pd.DataFrame(data.groupby(['product_id'])['income'].first()).reset_index()
    reduce_mem_usage(productGB)

    fig = px.bar(productGB, x = productGB.product_id, y = productGB.income, color = productGB.income)
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=1400, xaxis_title="Product ID", yaxis_title="Income", title_text="Boxplot to Visualize Income vs Product ID") 
    st.plotly_chart(fig)

    ##########################################################################################################################

    st.write('')
    st.write('')
    st.subheader('INCOME VS MONTHS')
    monthGB = pd.DataFrame(data.groupby(['month'])['income'].first()).reset_index()
    reduce_mem_usage(monthGB)

    fig = px.bar(monthGB, x = monthGB.month, y = monthGB.income, color = monthGB.income)
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=1000, xaxis_title="Month", yaxis_title="Income", title_text="Boxplot to Visualize Income vs Month") 
    st.plotly_chart(fig)

    ##########################################################################################################################

    st.write('')
    st.write('')
    st.subheader('REVENUE VS MONTHS')
    monthGB = pd.DataFrame(data.groupby(['month'])['revenue'].first()).reset_index()
    reduce_mem_usage(monthGB)

    fig = px.bar(monthGB, x = monthGB.month, y = monthGB.revenue, color = monthGB.revenue)
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=1000, xaxis_title="Month", yaxis_title="Revenue Generated", title_text="Boxplot to Visualize Revenue Generated Over Different Months") 
    st.plotly_chart(fig)

    ##########################################################################################################################

    st.write('Out of {} Total Observations, There are:'.format(data.shape[0]))
    st.write("{} Unique Product ID".format(len(data.product_id.unique())))

    st.write("{} Unique Store ID".format(len(data.store_id.unique())))

    st.write("{} Unique Customer Qualification ID".format(len(data.qualification.unique())))
    st.write("{} Unique Customer Age".format(len(data.age.unique())))

    st.write("{} Unique Store Groups".format(len(data.store_group.unique())))
    st.write("{} Unique Store Sizes".format(len(data.store_size.unique())))

    ##########################################################################################################################

    data.qualification = data.qualification.replace({'Degree or higher': 0, 
                                                    'GCSE': 1, 
                                                    'Other': 2, 
                                                    'None': 3, 
                                                    'Higher education': 4, 
                                                    'A Level': 5, 
                                                    'Unknown': 6})

    data.tenure = data.tenure.replace({'Owned outright': 0, 
                                    'Mortgaged': 1, 
                                    'Rented': 2, 
                                    'Other': 3, 
                                    'Unknown': 4})

    data.store_region = data.store_region.replace({'North': 0, 
                                                'London': 1, 
                                                'East': 2, 
                                                'Wales': 3, 
                                                'South': 4, 
                                                'Midlands': 5, 
                                                'Scotland': 6})


    ##########################################################################################################################


    # data = data[['month', 'product_id', 'store_id', 'net_spend', 'qualification', 'tenure', 'retired_flag', 'income','brand_id', 'store_region', 'revenue', 'K_flag']]
    data = sales.merge(products, on='product_id', how='left')
    reduce_mem_usage(data)

    ##########################################################################################################################

    st.write('')
    st.write('')
    st.subheader('NET SPEND VS MONTHS')
    spend = pd.DataFrame(data.groupby(['month'])['net_spend'].first()).reset_index()

    fig = px.bar(spend, x = spend.month, y = spend.net_spend, color = spend.month, barmode = 'group')
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=1000, xaxis_title="Month", yaxis_title="Net Spend", title_text="Net Spend over Different Months") 
    st.plotly_chart(fig)

    ##########################################################################################################################

    st.write('')
    st.write('')
    st.subheader('NET SPEND VS PRODUCT ID')
    productSpend = pd.DataFrame(data.groupby(['product_id'])['net_spend'].first()).reset_index()

    fig = px.bar(productSpend, x = productSpend.product_id, y = productSpend.net_spend, color = productSpend.product_id, barmode = 'group')
    fig.update_xaxes(rangeslider_visible=True, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=600, width=1400, xaxis_title="Month", yaxis_title="Net Spend", title_text="Net Spend over Different Product ID") 
    st.plotly_chart(fig)

    ##########################################################################################################################

    st.write('')
    st.write('')
    st.subheader('NET SPEND VS MONTHS W.R.T PRODUCT ID')
    productSpend = pd.DataFrame(data.groupby(['month', 'product_id'])['net_spend'].first()).reset_index()

    fig = px.bar(productSpend, x = productSpend.product_id, y = productSpend.net_spend, animation_frame = productSpend.month, color = productSpend.net_spend, barmode = 'group')
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=600, width=1400, xaxis_title="Month", yaxis_title="Net Spend", title_text="Net Spend For Different Months over a Course of Product ID's") 
    st.plotly_chart(fig)

    ##########################################################################################################################

    st.write('')
    st.write('')
    st.subheader('NET SPEND VS PRODUCT ID W.R.T MONTHS')
    productSpend = pd.DataFrame(data.groupby(['month', 'product_id'])['net_spend'].first()).reset_index()
    productSpend = productSpend[productSpend.net_spend < productSpend.net_spend.mean()]

    fig = px.bar(productSpend, x = productSpend.month, y = productSpend.net_spend, animation_frame = productSpend.product_id, color = productSpend.net_spend, barmode = 'group')
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=600, width=1000, xaxis_title="Month", yaxis_title="Net Spend", title_text="Net Spend For Different Product ID over a Course of Month") 
    st.plotly_chart(fig)

    ##########################################################################################################################