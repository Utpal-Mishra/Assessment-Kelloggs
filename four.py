import streamlit as st
import time

import sys
import numpy as np
import pandas as pd

from datetime import datetime as dt

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

def app():
    st.title("TRAIN DELAY ANALYSIS IN BELGIUM")
    
    # st.header("PART 4")
    
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
    # print("Data Shape: {}\n".format(data.shape))
    st.dataframe(data.head(10))
    
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

    st.write('Total No.of Trains: {}'.format(len(data.TRAIN_NO.unique())))

    ###########################################################################

    st.header("Visualising Data")

    trainsFrequency = pd.DataFrame(data.TRAIN_NO.value_counts()[:25]).reset_index()
    trainsFrequency.rename({'index': 'TRAIN_NO', 'TRAIN_NO': 'FREQUENCY'}, axis = 1, inplace = True)
    st.subheader("Train Data")
    st.write("Based on Group By Train Number")
    st.dataframe(trainsFrequency)

    st.subheader('Bar Plot')
    st.write('Frequency of Trains')
    fig = px.bar(trainsFrequency, x = trainsFrequency['TRAIN_NO'], y = trainsFrequency['FREQUENCY'], color = trainsFrequency['FREQUENCY'])
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=1600, xaxis_title="Train Numbers", yaxis_title="Frequency of Trains", title_text="Frequency of Trains") 
    # fig.show()
    st.plotly_chart(fig)

    st.write("Based on Group By Train Numbers with Features Representing Arrivals and Departures")
    trainNo              = data[['TRAIN_NO', 'DELAY_ARR', 'DELAY_DEP']]
    trainNo['DEP_STAT']  = trainNo.DELAY_DEP.apply(lambda x: 'Earlier Dep' if x>0 else 'Late Dep')
    trainNo['ARR_STAT']  = trainNo.DELAY_ARR.apply(lambda x: 'Earlier Arr' if x>0 else 'Late Arr')
    st.dataframe(trainNo.tail())

    st.subheader('Frequency of Trains at Arrivals')
    arr = pd.DataFrame(trainNo.groupby(['TRAIN_NO', 'ARR_STAT'])['DELAY_ARR'].first()).reset_index()

    fig = px.histogram(arr, x = arr.TRAIN_NO, y = arr.DELAY_ARR, color = arr.ARR_STAT,  barmode='group')
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=1200, xaxis_title="Train Numbers", yaxis_title="Frequency of Trains Numbers", title_text="Frequency of Trains Numbers in Berlin") 
    # fig.show()
    st.plotly_chart(fig)

    st.subheader('Frequency of Trains at Departures')
    dep = pd.DataFrame(trainNo.groupby(['TRAIN_NO', 'DEP_STAT'])['DELAY_DEP'].first()).reset_index()

    fig = px.histogram(dep, x = dep.TRAIN_NO, y = dep.DELAY_DEP, color = dep.DEP_STAT,  barmode='group')
    fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(height=500, width=1200, xaxis_title="Train Numbers", yaxis_title="Frequency of Trains Numbers", title_text="Frequency of Trains Numbers in Berlin") 
    # fig.show()
    st.plotly_chart(fig)

    st.write('[Notebook](https://github.com/Utpal-Mishra/Omdena-Berlin-Chapter-2023/blob/main/OmdenaBerlinChapter2023Part4.ipynb)')