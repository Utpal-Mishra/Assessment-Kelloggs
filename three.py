import streamlit as st
import time

import sys
import numpy as np
import pandas as pd

from datetime import datetime as dt
from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from streamlit_metrics import metric, metric_row

print('Folium Installed')
print('Libraries Imported')

sys.setrecursionlimit(100000)
#print("Installed Dependencies")

###########################################################################

def app():
    st.title("TRAIN DELAY ANALYSIS IN BELGIUM")
    
    # st.header("PART 3")
    
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
    st.write('Train Delay Data')
    st.dataframe(data.head(10))
    st.write("Data Shape: {}\n".format(data.shape))
    
    
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

    st.write('TRAIN DELAYS in ARRIVAL   : {} Seconds :: {} Hours'.format(round(data.groupby('DATDEP')['DELAY_ARR'].mean().sum(), 3), str(timedelta(seconds=data.groupby('DATDEP')['DELAY_ARR'].mean().sum()))))
    st.write('TRAIN DELAYS in DEPARTURE : {} Seconds :: {} Hours'.format(round(data.groupby('DATDEP')['DELAY_DEP'].mean().sum(), 3), str(timedelta(seconds=data.groupby('DATDEP')['DELAY_DEP'].mean().sum()))))

    delayARR = pd.DataFrame(data.groupby('DATDEP')['DELAY_ARR'].mean()).reset_index()
    st.subheader("Delay Data")
    st.write("Based on Group By Departure Date alongwith Delay in Arrival")
    st.dataframe(delayARR)

    st.subheader('Ddelay in Train Arrivals')
    fig = px.bar(delayARR, x = delayARR.DATDEP, y = delayARR.DELAY_ARR, color = delayARR.DELAY_ARR)
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=1200, xaxis_title="Date", yaxis_title="Train Delay (in Seconds)", title_text="DELAYS IN TRAIN ARRIVAL") 
    # fig.show()
    st.plotly_chart(fig)

    delayDEP = pd.DataFrame(data.groupby('DATDEP')['DELAY_DEP'].mean()).reset_index()

    st.subheader('Delay in Train Departures')
    fig = px.bar(delayDEP, x = delayDEP.DATDEP, y = delayDEP.DELAY_DEP, color = delayDEP.DELAY_DEP)
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=1200, xaxis_title="Date", yaxis_title="Train Delay (in Seconds)", title_text="DELAYS IN TRAIN DEPARTURE") 
    # fig.show()
    st.plotly_chart(fig)
        
    st.write('Average Time Delay')
    st.dataframe(pd.DataFrame(data.groupby('DATDEP')['DELAY_ARR'].mean()).reset_index())

    st.write('[Notebook](https://github.com/Utpal-Mishra/Omdena-Berlin-Chapter-2023/blob/main/OmdenaBerlinChapter2023Part3.ipynb)')