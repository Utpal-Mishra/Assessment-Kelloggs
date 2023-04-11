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
    
    # st.header("PART 2")
    
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
    st.write("Preparing New Data with Attributes: ['DELAY_DEP', 'DELAY_ARR']")

    delay = data[['DELAY_DEP', 'DELAY_ARR']]
    delay['DEP_STAT']  = delay.DELAY_DEP.apply(lambda x: 'Earlier Dep' if x>0 else 'Late Dep')
    delay['DELAY_DEP'] = delay.DELAY_DEP.apply(lambda x: abs(x))
    delay['ARR_STAT']  = delay.DELAY_ARR.apply(lambda x: 'Earlier Arr' if x>0 else 'Late Arr')
    delay['DELAY_ARR'] = delay.DELAY_ARR.apply(lambda x: abs(x))

    st.subheader("Delay Data")
    st.write("Based on Early/ Late Arrivals/ Departures")
    st.dataframe(delay.head(10))

    dep = pd.DataFrame(delay.groupby(['DEP_STAT'])['DELAY_DEP'].mean()).reset_index()
    arr = pd.DataFrame(delay.groupby(['ARR_STAT'])['DELAY_ARR'].mean()).reset_index()

    st.subheader("Percentage of Earlier Arrival vs Late Delays via Berlin Train Station")
    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=dep.DEP_STAT, values=dep.DELAY_DEP, hole=.5, pull=[0, 0.2], name="Departures"), 1, 1)
    fig.add_trace(go.Pie(labels=arr.ARR_STAT, values=arr.DELAY_ARR, hole=.5, pull=[0, 0.2], name="Arrivals"), 1, 2)

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")

    fig.update_layout(
        height=500, width=1200,
        title_text="Percentage of Earlier Arrival vs Late Delays via Berlin Train Station",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='DEP', x=0.20, y=0.5, font_size=20, showarrow=False),
                    dict(text='ARR', x=0.80, y=0.5, font_size=20, showarrow=False)])
    # fig.show()
    st.plotly_chart(fig)

    st.write('[Notebook](https://github.com/Utpal-Mishra/Omdena-Berlin-Chapter-2023/blob/main/OmdenaBerlinChapter2023Part2.ipynb)')