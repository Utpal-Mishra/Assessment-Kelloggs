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
from streamlit_folium import folium_static

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
    
    # st.header("PART 6")
    
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

    data['ORIGIN']      = data.RELATION_DIRECTION.apply(lambda x: x.split(':')[1].split(' ->')[0])
    data['DESTINATION'] = data.RELATION_DIRECTION.apply(lambda x: x.split(':')[1].split('> ')[1])
    data['WEEKDAY']     = data['REAL_DATE_ARR'].apply(lambda x: x.strftime('%A'))

    st.dataframe(data.tail())

    # st.write('Total No.of Trains: {}'.format(len(data.TRAIN_NO.unique())))

    ###########################################################################
        
    mapData = pd.DataFrame(data.RELATION_DIRECTION.unique())
    mapData['ORIGIN']      = mapData[0].apply(lambda x: x.split(':')[1].split(' ->')[0])
    mapData['DESTINATION'] = mapData[0].apply(lambda x: x.split(':')[1].split('> ')[1])
    del mapData[0]

    st.write('Mapping Data')
    st.dataframe(data.head())
    
    ########################################################################### 

    Arr = []
    Arr_Long = []
    Arr_Lat = []
    Dep = []
    Dep_Long = []
    Dep_Lat = []

    for i in data.ORIGIN.unique():
      try:
        geolocator = Nominatim(user_agent="four_square")
        location = geolocator.geocode(i)
        latitude = location.latitude
        longitude = location.longitude
        Dep.append(i)
        Dep_Long.append(longitude)
        Dep_Lat.append(latitude)
      except:
        print(i+" Error")

    for i in data.DESTINATION.unique():
      try:
        geolocator = Nominatim(user_agent="four_square")
        location = geolocator.geocode(i)
        latitude = location.latitude
        longitude = location.longitude
        Arr.append(i)
        Arr_Long.append(longitude)
        Arr_Lat.append(latitude)
      except:
        print(i+" Error")

    ########################################################################### 
     
    Dep_Cord = pd.DataFrame({"ORIGIN": Dep, "Dep_Long": Dep_Long, "Dep_Lat": Dep_Lat})
    Arr_Cord = pd.DataFrame({"DESTINATION": Arr, "Arr_Long": Arr_Long, "Arr_Lat": Arr_Lat})

    mapData = pd.merge(mapData, Dep_Cord, on='ORIGIN', how='outer')
    mapData = pd.merge(mapData, Arr_Cord, on='DESTINATION', how='outer')
    mapData.dropna(inplace=True)
    # print("Dimensions of New Data: ", mapData.shape)
    st.write('Adding Latitudes and Longitudes to the Mapping Data')
    st.dataframe(mapData.head())

    ########################################################################### 

    st.subheader('Folium Map')

    address = 'Belgium'

    geolocator = Nominatim(user_agent="four_square")
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    st.write('The geograpical coordinate of Belgium are {}, {}.'.format(latitude, longitude))

    map = folium.Map(location = [latitude, longitude], zoom_start = 14) # tiles = 'Stamen Terrain')
    # map
    # folium_static(map)

    ########################################################################### 

    incidents = folium.map.FeatureGroup()
    incidents.add_child(folium.CircleMarker([latitude, longitude], 
                                                radius = 5, 
                                                color = 'red', 
                                                fill_color = 'red'))
    map.add_child(incidents)
    folium.Marker([latitude, longitude], popup = 'Berlin').add_to(map)
    # map
    folium_static(map)

    ########################################################################### 

    st.subheader('Folium Map with Train Stations Connected to Berlin')

    map = folium.Map(location = [latitude, longitude], zoom_start = 6) # tiles = 'Stamen Terrain')

    incidents = folium.map.FeatureGroup()

    # Berlin label
    incidents.add_child(folium.CircleMarker([latitude, longitude], 
                                                radius = 5, 
                                                color = 'red', 
                                                fill_color = 'red'))
    map.add_child(incidents)
    folium.Marker([latitude, longitude], popup = 'Berlin').add_to(map)

    incidents = folium.map.FeatureGroup()

    # loop through the origin stations and add each to the incidents feature group
    for lat, lng, in zip(mapData.Dep_Lat, mapData.Dep_Long):
        incidents.add_child(
            folium.CircleMarker(
                [lat, lng],
                radius=5, # define how big you want the circle markers to be
                color='yellow',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6
            )
        )
      
    # add pop-up text to each marker on the map
    latitudes = list(mapData.Dep_Lat)
    longitudes = list(mapData.Dep_Long)
    labels = list(mapData.ORIGIN)

    for lat, lng, label in zip(latitudes, longitudes, labels):
        folium.Marker([lat, lng], popup=label).add_to(map)    
        
    # add incidents to map
    map.add_child(incidents)
    # map
    folium_static(map)

    ########################################################################### 

    st.subheader('Folium Map with Train Network')

    map = folium.Map(location = [latitude, longitude], zoom_start = 6)

    Dep_Longitude = mapData.Dep_Long.astype(float).tolist()
    Dep_Latitude  = mapData.Dep_Lat.astype(float).tolist()

    Points = []

    # read a series of points from coordinates and assign them to points object
    for i in range(len(Dep_Latitude)):
        Points.append([Dep_Latitude[i], Dep_Longitude[i]])

    # specify an icon of your desired shape or chosing in place for the coordinates points
    for index, lat in enumerate(Dep_Latitude):
        folium.Marker([lat, 
                      Dep_Longitude[index]],
                      popup=('Train Station: {} \n '.format(index)),
                      icon = folium.Icon(color = 'blue', icon_color = 'white', prefix = 'fa', icon = 'train')).add_to(map)
    
    folium.PolyLine(Points, color = 'red', weight = 3, opacity = 1,
                tooltip = 'Trains Route').add_to(map)
    
    folium_static(map)

    ########################################################################### 