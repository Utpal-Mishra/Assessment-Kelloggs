import streamlit as st
import time

import sys
import numpy as np # library to handle data in a vectorized manner
import pandas as pd

import random # library for random number generation
from datetime import datetime as dt

import seaborn as sns # plotting library
import matplotlib.cm as cm # plotting library
import matplotlib.colors as colors
import matplotlib.pyplot as plt
# backend for rendering plots within the browser
# %matplotlib inline 
import plotly.express as px  # plotting library
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import requests # library to handle requests
import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#!conda install -c conda-forge geopy --yes 
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 
    
# tranforming json file into a pandas dataframe library
#from pandas.io.json import json_normalize
from pandas import json_normalize

#!conda install -c conda-forge folium=0.5.0 --yes
import folium # plotting library

print('Folium Installed')
print('Libraries Imported')

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
#from pandas.io.json import json_normalize
from pandas import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

print('Libraries imported.')

sys.setrecursionlimit(100000)
#print("Installed Dependencies")


def app():
    st.title("TRAIN DELAY ANALYSIS IN BELGIUM")
    
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
    
    # Air Transport
    path = 'AirTransport.xlsx'
    air = pd.read_excel(path)
    # print("Air Transport Data Shape: {}\n".format(air.shape))
    air.head()

    # Fright Transport
    path = 'Roads.xlsx'
    roads = pd.read_excel(path)
    # print("Freight (Roads) Data Shape: {}\n".format(roads.shape))
    roads.head()

    path = 'Railways.xlsx'
    rail = pd.read_excel(path)
    # print("Freight (Rail) Data Shape: {}\n".format(rail.shape))
    rail.head()

    path = 'InlandWaterways.xlsx'
    water = pd.read_excel(path)
    water.fillna(0)
    # print("Freight (Inland Waterways) Data Shape: {}\n".format(water.shape))
    water.head()

    M1 = roads.groupby('Countries').sum().add(rail.groupby('Countries').sum(), fill_value=0).reset_index()
    M2 = M1.groupby('Countries').sum().add(water.groupby('Countries').sum(), fill_value=0).reset_index()
    
    # Fright Transport
    path = 'MotorCoaches.xlsx'
    coach = pd.read_excel(path)
    # print("Freight (Roads) Data Shape: {}\n".format(coach.shape))
    coach.head()

    path = 'PassengerCars.xlsx'
    car = pd.read_excel(path)
    # print("Freight (Rail) Data Shape: {}\n".format(car.shape))
    car.head()

    path = 'Trains.xlsx'
    train = pd.read_excel(path)
    # print("Freight (Inland Waterways) Data Shape: {}\n".format(train.shape))
    train.head()

    M3 = coach.groupby('Countries').sum().add(car.groupby('Countries').sum()).reset_index()
    M4 = M3.groupby('Countries').sum().add(train.groupby('Countries').sum()).reset_index()

    st.write('ABOUT DATA')
    st.write("""The initial approach is to examine the signifiance of widely used transportation channels - Air, Freight and Inland - in/ via Berlin and the engagement of the commuters on a day to day basis.
                The extracted data comprises details collected from more than a decade of records throughout the Europian Union.""")

    # if st.checkbox("Show DataFrame"):    
    # st.dataframe(air) # data

    st.subheader("Exploratory Data Analysis")

    ##########################################################################################################################

    if st.checkbox("Air Transport"):
    
        air = air.rename(columns = {'TIME': 'Countries'})
        # air.tail()
        st.write('Air Transport Data')
        st.dataframe(air)
        st.write("Data Shape: {}\n".format(air.shape))

        air = air.melt(id_vars=["Countries"], var_name = "Year", value_name = "Passengers Frequency")
        air.sort_values(["Countries", "Year"], inplace = True)
        # air.head()
        st.write('Formatting the Respective Data - More than Year Columns into Rows')
        st.dataframe(air)
        
        st.write("Plotting Bar Plot to Analyse Frequency of Passengers Traveling Across Countries via Air Transport")  
        
        # Bar Plot
        # if st.checkbox("Frequency of Passengers Traveling"): 
        fig = px.bar(air, x="Countries", y="Passengers Frequency", animation_frame="Year", color="Countries", barmode="group")
        fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_layout(height=600, width=1400, plot_bgcolor="black", title_text="Frequency of Passengers Traveling Across Countries via Air Transport")
        # fig.show()
        st.plotly_chart(fig)

    ##########################################################################################################################

    if st.checkbox("Freight Transport"):
    
        roads = roads.rename(columns = {'TIME': 'Countries'})
        # roads.tail()
        st.write('Freight Transport Data - Roads')
        st.dataframe(roads)
        st.write("Data Shape: {}\n".format(roads.shape))

        roads = roads.melt(id_vars=["Countries"], var_name = "Year", value_name = "Passengers Frequency")
        roads.sort_values(["Countries", "Year"], inplace = True)
        # roads.head()
        st.write('Formatting the Respective Data - More than Year Columns into Rows')
        st.dataframe(roads)

        st.write('Frequency of Passengers Traveling Across Countries via Freight Transport - Roads')
      
        # Bar Plot
        # if st.checkbox("Frequency of Passengers Traveling"): 
        fig = px.bar(roads, x="Countries", y="Passengers Frequency", animation_frame="Year", color="Countries", barmode="group")
        fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_layout(height=600, width=1400, plot_bgcolor="black", title_text="Frequency of Passengers Traveling Across Countries via Freight Transport - Roads")
        # fig.show()
        st.plotly_chart(fig)
        
        ##########################################################################################

        rail = rail.rename(columns = {'TIME': 'Countries'})
        # rail.tail()
        st.write('Freight Transport Data - Rail')
        st.dataframe(rail)
        st.write("Data Shape: {}\n".format(rail.shape))

        rail = rail.melt(id_vars=["Countries"], var_name = "Year", value_name = "Passengers Frequency")
        rail.sort_values(["Countries", "Year"], inplace = True)
        # rail.head()
        st.write('Formatting the Respective Data - More than Year Columns into Rows')
        st.dataframe(rail)

        st.write('Frequency of Passengers Traveling Across Countries via Freight Transport - Rail')

        # Bar Plot
        # if st.checkbox("Frequency of Passengers Traveling"): 
        fig = px.bar(rail, x="Countries", y="Passengers Frequency", animation_frame="Year", color="Countries", barmode="group")
        fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_layout(height=600, width=1400, plot_bgcolor="black", title_text="Frequency of Passengers Traveling Across Countries via Freight Transport - Rail")
        # fig.show()
        st.plotly_chart(fig)  

        ##########################################################################################

        water = water.rename(columns = {'TIME': 'Countries'})
        # water.tail()
        st.write('Freight Transport Data - Water')
        st.dataframe(water)
        st.write("Data Shape: {}\n".format(water.shape))

        water = water.melt(id_vars=["Countries"], var_name = "Year", value_name = "Passengers Frequency")
        water.sort_values(["Countries", "Year"], inplace = True)
        # water.head()
        st.write('Formatting the Respective Data - More than Year Columns into Rows')
        st.dataframe(water)

        st.write('Frequency of Passengers Traveling Across Countries via Freight Transport - Water')

        # Bar Plot
        # if st.checkbox("Frequency of Passengers Traveling"): 
        fig = px.bar(water, x="Countries", y="Passengers Frequency", animation_frame="Year", color="Countries", barmode="group")
        fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_layout(height=600, width=1400, plot_bgcolor="black", title_text="Frequency of Passengers Traveling Across Countries via Freight Transport - Waterways")
        # fig.show()
        st.plotly_chart(fig)

        ##########################################################################################

        st.write('Orthographics Plot on Frequency of Passengers Traveling Across Countries via Freight')

        data = M2.melt(id_vars=["Countries"], var_name = "Year", value_name = "Passengers Frequency")
        data.sort_values(["Countries", "Year"], inplace = True)
        st.write('Freight Transport Data')
        st.dataframe(data.head())
        st.write("Data Shape: {}\n".format(data.shape))

        Country = []
        Longitude = []
        Latitude = []

        for i in data.Countries.unique():
            geolocator = Nominatim(user_agent="four_square")
            location = geolocator.geocode(i)
            latitude = location.latitude
            longitude = location.longitude
            Country.append(i)
            Longitude.append(longitude)
            Latitude.append(latitude)
        # print(i, longitude, latitude)

        # data["Longitude"] = Longitude
        # data["Latitude"] = Latitude

        Coordinates = pd.DataFrame({"Countries": Country,
                    "Longitude": Longitude,
                    "Latitude": Latitude})

        # Coordinates.head()
        newdata = pd.merge(data, Coordinates, on='Countries', how='outer')
        newdata['Passengers Frequency'] = newdata['Passengers Frequency'].apply(lambda x: round(x, 2))
        print("Dimensions of New Data: ", newdata.shape)

        fig = px.scatter_geo(newdata, 
                            lat='Latitude', 
                            lon='Longitude', 
                            size='Passengers Frequency', 
                            animation_frame="Year", 
                            #animation_group = "Specie",
                            title='Frequency of Passengers Traveling Across Countries via Freight', 
                            hover_name="Countries",
                            projection = "orthographic", 
                            width = 1400,
                            height = 700, 
                            color = "Countries")
        fig.update(layout_coloraxis_showscale=True)
        # fig.show()
        st.plotly_chart(fig)

    ##########################################################################################################################

    if st.checkbox("Inland"):
    
        coach = coach.rename(columns = {'TIME': 'Countries'})
        # air.tail()
        st.write('Inland Transport Data - Motor Coaches')
        st.dataframe(coach)
        st.write("Data Shape: {}\n".format(coach.shape))

        coach = coach.melt(id_vars=["Countries"], var_name = "Year", value_name = "Passengers Frequency")
        coach.sort_values(["Countries", "Year"], inplace = True)
        #coach.head()
        st.write('Formatting the Respective Data - More than Year Columns into Rows')
        st.dataframe(coach)

        st.write('Frequency of Passengers Traveling Across Countries via Inland Transport- Motor Coaches')
        
        # Bar Plot
        fig = px.bar(coach, x="Countries", y="Passengers Frequency", animation_frame="Year", color="Countries", barmode="group")
        fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_layout(height=600, width=1400, plot_bgcolor="black", title_text="Frequency of Passengers Traveling Across Countries via Inland Transport - Motor Coaches")
        # fig.show()
        st.plotly_chart(fig)
       
        ##########################################################################################

        car = car.rename(columns = {'TIME': 'Countries'})
        # car.tail()
        st.write('Inland Transport Data - Passenger Cars')
        st.dataframe(car)
        st.write("Data Shape: {}\n".format(car.shape))

        car = car.melt(id_vars=["Countries"], var_name = "Year", value_name = "Passengers Frequency")
        car.sort_values(["Countries", "Year"], inplace = True)
        #car.head()
        st.write('Formatting the Respective Data - More than Year Columns into Rows')
        st.dataframe(car)

        st.write('Frequency of Passengers Traveling Across Countries via Inland Transport - Passenger Cars')

        # Bar Plot
        fig = px.bar(car, x="Countries", y="Passengers Frequency", animation_frame="Year", color="Countries", barmode="group")
        fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_layout(height=600, width=1400, plot_bgcolor="black", title_text="Frequency of Passengers Traveling Across Countries via Inland Transport - Passenger Cars")
        # fig.show()
        st.plotly_chart(fig)

        ##########################################################################################      

        train = train.rename(columns = {'TIME': 'Countries'})
        # train.tail()
        st.write('Inland Transport Data - Trains')
        st.dataframe(train)
        st.write("Data Shape: {}\n".format(train.shape))

        train = train.melt(id_vars=["Countries"], var_name = "Year", value_name = "Passengers Frequency")
        train.sort_values(["Countries", "Year"], inplace = True)
        train.head()
        st.write('Formatting the Respective Data - More than Year Columns into Rows')
        st.dataframe(train)

        st.write('Frequency of Passengers Traveling Across Countries via Inland Transport - Train')

        # Bar Plot
        fig = px.bar(train, x="Countries", y="Passengers Frequency", animation_frame="Year", color="Countries", barmode="group")
        fig.update_xaxes(rangeslider_visible=False, showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_layout(height=600, width=1400, plot_bgcolor="black", title_text="Frequency of Passengers Traveling Across Countries via Inland Transport - Train")
        # fig.show()
        st.plotly_chart(fig)

    st.write('[Notebook](https://github.com/Utpal-Mishra/Omdena-Berlin-Chapter-2023/blob/main/OmdenaBerlinChapter2023Part1.ipynb)')