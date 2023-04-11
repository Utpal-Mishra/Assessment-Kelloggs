import streamlit as st
import time

import sys
import numpy as np # library to handle data in a vectorized manner
import pandas as pd # library for data analsysis
# from pandas.io.json import json_normalize
# tranforming json file into a pandas dataframe library
# from pandas.io.json import json_normalize
from pandas import json_normalize # tranform JSON file into a pandas dataframe

import requests # library to handle requests
import random # library for random number generation
from datetime import datetime as dt

import seaborn as sns # plotting library
# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
# backend for rendering plots within the browser
# %matplotlib inline 
import plotly.express as px  # plotting library
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files
#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
    
#!conda install -c conda-forge folium=0.5.0 --yes
import folium # plotting library

print('Folium Installed')
print('Libraries Imported')

sys.setrecursionlimit(100000)
#print("Installed Dependencies")

def app():
    st.title("TRAIN DELAY ANALYSIS IN BELGIUM")
    
    # st.header("PART 5")
    
    st.subheader("Loading Page....")
    
    label = st.empty()
    bar = st.progress(0)
    
    for i in range(100):
        # Update progress bar with iterations
        label.text(f'Loaded {i+1} %')
        bar.progress(i+1)
        time.sleep(0.01)
    
    ".... and now we're done!!!"
    
    # Air Transport
    path = 'Data_raw_punctuality_202301.csv'
    data = pd.read_csv(path)
    st.write("Data Shape: {}\n".format(data.shape))
    st.dataframe(data.head())
    
    
    ###########################################################################
    
    # data.info()
    # data.isnull().sum()
    st.header("Exploratory Data Analysis")

    st.write("Dropping Redundant Feature Variables: ['RELATION_DIRECTION', 'REAL_TIME_ARR', 'LINE_NO_DEP', 'LINE_NO_ARR']")
    # data.drop(columns = ['LINE_NO_DEP', 'LINE_NO_ARR'], axis=1, inplace=True)
    data.dropna(subset = ['RELATION_DIRECTION', 'REAL_TIME_ARR', 'LINE_NO_DEP', 'LINE_NO_ARR'], inplace = True)
    st.dataframe(data.head())

    st.subheader("Checking Null Values")
    st.dataframe(data.isnull().sum())
    st.write('Total Null Values: {}'.format(data.isnull().sum().sum()))

    st.subheader("Formating Date and Time Entries")
    #
    data.REAL_TIME_ARR    = pd.to_datetime(data.REAL_TIME_ARR, format = '%H:%M:%S').dt.time
    data.REAL_TIME_DEP    = pd.to_datetime(data.REAL_TIME_DEP, format = '%H:%M:%S').dt.time
    data.PLANNED_TIME_ARR = pd.to_datetime(data.PLANNED_TIME_ARR, format = '%H:%M:%S').dt.time
    data.PLANNED_TIME_DEP = pd.to_datetime(data.PLANNED_TIME_DEP, format = '%H:%M:%S').dt.time

    #
    data.DATDEP           = pd.to_datetime(data.DATDEP, format = '%d%b%Y')
    data.PLANNED_DATE_ARR = pd.to_datetime(data.PLANNED_DATE_ARR, format = '%d%b%Y')
    data.PLANNED_DATE_DEP = pd.to_datetime(data.PLANNED_DATE_DEP, format = '%d%b%Y')
    data.REAL_DATE_ARR    = pd.to_datetime(data.REAL_DATE_ARR, format = '%d%b%Y')
    data.REAL_DATE_DEP    = pd.to_datetime(data.REAL_DATE_DEP, format = '%d%b%Y')

    #
    data.TRAIN_NO   = pd.Categorical(data.TRAIN_NO)
    data.RELATION   = pd.Categorical(data.RELATION)
    data.TRAIN_SERV = pd.Categorical(data.TRAIN_SERV)

    st.dataframe(data.tail())

    # st.write('Total No.of Trains: {}'.format(len(data.TRAIN_NO.unique())))

    ###########################################################################

    st.header("Visualising Data")

    railOp              = data[['TRAIN_SERV', 'DELAY_ARR', 'DELAY_DEP']]
    railOp['DEP_STAT']  = railOp.DELAY_DEP.apply(lambda x: 'Earlier Dep' if x>0 else 'Late Dep')
    railOp['ARR_STAT']  = railOp.DELAY_ARR.apply(lambda x: 'Earlier Arr' if x>0 else 'Late Arr')
    st.subheader("Train Operator Data")
    st.write("Based on Group By Train Operator with Features Representing Arrivals and Departures")
    st.dataframe(railOp)
        
    arr = pd.DataFrame(railOp.groupby(['TRAIN_SERV', 'ARR_STAT'])['DELAY_ARR'].first()).reset_index()

    st.subheader('Frequency of Trains Operators at Arrivals')
    fig = px.histogram(arr, x = arr.TRAIN_SERV, y = arr.DELAY_ARR, color = arr.ARR_STAT,  barmode='group')
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=700, xaxis_title="Train Operators", yaxis_title="Frequency of Trains Operators", title_text="Frequency of Trains Operators in Berlin") 
    # fig.show()
    st.plotly_chart(fig)
        
    dep = pd.DataFrame(railOp.groupby(['TRAIN_SERV', 'DEP_STAT'])['DELAY_DEP'].first()).reset_index()

    st.subheader('Frequency of Trains Operators at Departures')
    fig = px.histogram(dep, x = dep.TRAIN_SERV, y = dep.DELAY_DEP, color = dep.DEP_STAT,  barmode='group')
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=700, xaxis_title="Train Operators", yaxis_title="Frequency of Trains Operators", title_text="Frequency of Trains Operators in Berlin") 
    # fig.show()
    st.plotly_chart(fig)

    st.write('[Notebook](https://github.com/Utpal-Mishra/Omdena-Berlin-Chapter-2023/blob/main/OmdenaBerlinChapter2023Part5.ipynb)')