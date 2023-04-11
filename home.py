import streamlit as st

def app():
        
    st.title("OMDENA BERLIN CHAPTER 2021")
    
    video = open('Berlin.mp4', 'rb')
    videoFile = video.read()

    st.video(videoFile, format="video/mp4", start_time=0)

    st.write("")
    st.write("")
    
    st.header("TRAIN DELAY ANALYSIS IN BELGIUM")
    
    st.write("")
    st.write("")
    
    st.write("APPLICATION DESCRIPTION")
    
    st.write("")
    
    st.write("Page 1: About Data")
    
    st.write("")
    
    st.write("Page 2: About Delay Proportions in Arrival and Departures")
    
    st.write("")
    
    st.write("Page 3: About Delay via Date")
    
    st.write("")
    
    st.write("Page 4: About Delay via Train No.")
    
    st.write("")
    
    st.write("Page 5: About Delay via Railway Operator")
    
    st.write("")
    
    st.write("Page 6: Mapping Train Delay")
    
    st.write("")
    
    st.write("Page 7: About Train Status")
    
    