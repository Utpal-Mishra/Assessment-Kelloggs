import streamlit as st
import time

import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import plotly.express as px
from datetime import datetime as dt

from streamlit_metrics import metric, metric_row


sys.setrecursionlimit(100000)
#print("Installed Dependencies")

def app():
    st.title("COVID-19 BEFORE-DURING ANALYSIS IN FRANCE")
    
    st.header("LOCKDOWN PHASES")
    
    st.subheader("Loading the COVID-19 Data....")
    
    file = st.file_uploader("Upload file")
    show_file = st.empty()
    
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join([".csv", ".xls", ".xlsx"]))
        return
    
    label = st.empty()
    bar = st.progress(0)
    
    for i in range(100):
        # Update progress bar with iterations
        label.text(f'Loaded {i+1} %')
        bar.progress(i+1)
        time.sleep(0.01)
    
    path = file #'new_master_data.csv'
    data = pd.read_csv(path)
    #print("Data Shape: ", data.shape)
    data.head()
    #print("Data Shape: ", data.shape)
    #data.head()
    
    ".... and now we're done!!!"
    
    
    st.subheader("Visualize and Compare Pollutants and CO2 Sources in All Lockdown Phases")
    
    
    pollutants = data[["date", "City", "driving", "transit", "walking"]]
    
    pollutants = pollutants.melt(id_vars=["date", "City"], var_name = "CO2 Sources", value_name = "Concentration")
    pollutants.sort_values(["date", "CO2 Sources"], inplace = True)
    
    
    def barplot(data, x, y, frame, color, ylabel, title):
      fig = px.bar(pollutants, x=x, y=y, animation_frame = frame, color=color)
      fig.update_xaxes(title_text = "France Cities", rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
      fig.update_yaxes(title_text = ylabel, showline=True, linewidth=2, linecolor='black', mirror=True)
    
      fig.add_trace(go.Scatter(x = ['2020-03-01', '2020-04-15', '2020-07-20', '2020-11-15', '2021-01-20', '2021-03-27', '2021-06-15'], y = [800, 800, 800, 800, 800, 800, 800], 
                               mode="text", name="Labels", text=["No Lockdown", "1st Lockdown", "No Lockdown", "2nd Lockdown", "No Lockdown", "3rd Lockdown", "No Lockdown"], textposition="top center"))
      fig.update_xaxes(title_text = "Date", rangeslider_visible=True, showline=True, linewidth=2, linecolor='black', mirror=True)
      fig.update_yaxes(title_text = "Average Concentration", showline=True, linewidth=2, linecolor='black', mirror=True)
      fig.update_layout(height=700, width=1500, title_text="Average Concentration (Unit - µg/m³) of Air Pollutants in France from 2020-21 ",
                          shapes = [dict(type = "rect", y0 = -50, y1 = 1200, x0="2020-02-05", x1="2020-03-17", name = "No Lockdown", fillcolor = "green", opacity = 0.5),
                                    dict(type = "rect", y0 = -50, y1 = 1200, x0="2020-03-17", x1="2020-05-10", name = "1st Lockdown", fillcolor = "red", opacity = 0.5),
                                    dict(type = "rect", y0 = -50, y1 = 1200, x0="2020-05-10", x1="2020-10-17", name = "No Lockdown", fillcolor = "green", opacity = 0.5), 
                                    dict(type = "rect", y0 = -50, y1 = 1200, x0="2020-10-17", x1="2020-12-14", name = "2nd Lockdown", fillcolor = "red", opacity = 0.5),
                                    dict(type = "rect", y0 = -50, y1 = 1200, x0="2020-12-14", x1="2021-02-26", name = "No Lockdown", fillcolor = "green", opacity = 0.5),
                                    dict(type = "rect", y0 = -50, y1 = 1200, x0="2021-02-26", x1="2021-05-02", name = "3rd Lockdown", fillcolor = "red", opacity = 0.5),
                                    dict(type = "rect", y0 = -50, y1 = 1200, x0="2021-05-02", x1="2021-07-27", name = "No Lockdown", fillcolor = "green", opacity = 0.5)]) 
      #fig.show()
      st.plotly_chart(fig)
      
    
    if st.checkbox("Compare and Analyse CO2 Sources through the Lockdown Phases"):
        barplot(pollutants, 
                "date", 
                "Concentration", 
                "City", 
                "CO2 Sources", 
                "Average Concentration of CO2 Sources (Unit - µg/m³)", 
                "Air Pollutants in France before-ongoing COVID-19")
      
        
    
    ###########################################################################
    
    
    
    pollutants = data[["date", "City", 
                       "('median', 'co')", "('median', 'dew')", "('median', 'no2')", "('median', 'o3')", "('median', 'pm10')", "('median', 'pm25')", "('median', 'so2')"]]
    
    pollutants = pollutants.melt(id_vars=["date", "City"], var_name = "Pollutants", value_name = "Concentration")
    pollutants.sort_values(["date", "Pollutants"], inplace = True)
     
      
    if st.checkbox("Compare and Analyse Pollutants through the Lockdown Phases"):
        barplot(pollutants, 
                "date", 
                "Concentration", 
                "City", 
                "Pollutants", 
                "Average Concentration of CO2 Sources (Unit - µg/m³)", 
                "Air Pollutants in France before-ongoing COVID-19")
    
    
    
    ##############################################################################
    
    
    
    pollutants = data[["date", "City", "transit", "walking",
                       "('median', 'co')", "('median', 'dew')", "('median', 'no2')", "('median', 'o3')", "('median', 'pm10')", "('median', 'pm25')", "('median', 'so2')"]]
    
    pollutants = pollutants.melt(id_vars=["date", "City"], var_name = "Pollutants", value_name = "Concentration")
    pollutants.sort_values(["date", "Pollutants"], inplace = True)
      
      
    if st.checkbox("Compare and Analyse Pollutants with CO2 Sources through the Lockdown Phases"):
        barplot(pollutants, 
                "date", 
                "Concentration", 
                "City", 
                "Pollutants", 
                "Average Concentration of CO2 Sources (Unit - µg/m³)", 
                "Air Pollutants in France before-ongoing COVID-19")
        

    ########################################################################### 