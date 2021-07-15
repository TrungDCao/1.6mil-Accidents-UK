#!/usr/bin/env python
# coding: utf-8

# ## Mapping

# In[5]:


dict_week = {1:"Sunday",2:"Monday",3:"Tuesday",4:"Wednesday",5:"Thursday",6:"Friday",7:"Saturday"}

dict_road_type = {1:"Roundabout", 2:"One way street", 3:"Dual carriageway",
                                                         6:"Single carriageway", 7:"Slip road", 9:"Unknown",
                                                         12:"One way street/Slip road", -1:"Missing data"}

dict_road_class = {1:"Motorway",2:"A(M)",3:"A",4:"B",5:"C", 6:"Unclassified"} 


dict_pd_human_control = {0:"None within 50 metres",1:"Control by school crossing patrol",2:"Control by other authorised person",
                      -1:"Data missing"}

                                                                                        
dict_junction_control =  {0:"Not at junction or within 20 metres",1:"Authorised person",2:"Auto traffic signal",3:"Stop sign",4:"Give way or uncontrolled",-1:"Data missing or out of range"}


dict_light_condition = {1:"Daylight", 4:"Darkness - lights lit", 5:"Darkness - lights unlit",
                                                                     6:"Darkness - no lighting", 7:"Darkness - lighting unknown",
                                                                     -1:"Data missing"}
                                                          

dict_weather_condition = {1:"Fine no high winds", 2:"Raining no high winds", 3:"Snowing no high winds",
                                                                         4:"Fine + high winds", 5:"Raining + high winds", 6:"Snowing + high winds",
                                                                         7:"Fog or mist", 8:"Other", 9:"Unknown", -1:"Data missing"}
                                                        
dict_road_surface = {1:"Dry", 2:"Wet or damp", 3:"Snow", 4:"Frost or ice", 5:"Flood over 3cm. deep", 6:"Oil or diesel", 7:"Mud",
                                                                                   -1:"Data missing"}
                                                      
dict_special_conditions =  {0:"None", 1:"Auto traffic signal - out", 2:"Auto signal part defective",
                                                                                         3:"Road sign or marking defective or obscured",
                                                                                         4:"Roadworks", 5:"Road surface defective", 6:"Oil or diesel",
                                                                                         7:"Mud", -1:"Data missing"}
                                                                       
dict_hazards = {0:"None",
                           1:"Vehicle load on road", 2:"Other object on road", 3:"Previous accident",
                           4:"Dog on road", 5:"Other animal on road", 
                           6:"Pedestrian in carriageway - not injured",
                           7:"Any animal in carriageway (except ridden horse)",
                           -1:"Data missing"}
dict_urban_rural = {1:"Urban", 2:"Rural", 3:"Unallocated"}


# # Application

# In[6]:


import streamlit as st
import pickle
import pandas as pd
import pandas as pd
#import dataset for moddling but withour normalisation
accidentsNotNorm = pd.read_csv(r"C:\Users\DETCAO03\V-Case study\02_Dataset\Used\Cleaned_not_normalized.csv",low_memory=False, encoding='utf-8')


# In[7]:


def map(arg, dicti):
    dicti = dict((y,x) for x,y in dicti.items())
    arg = pd.Series(arg)
    arg = float(arg.map(lambda x: dicti.get(x,x)))
    return arg


# In[11]:


#load models
lin_model=pickle.load(open('lin_model.pkl','rb'))
log_model=pickle.load(open('log_model.pkl','rb'))
dt_model=pickle.load(open('dt_model.pkl','rb'))
nb_model=pickle.load(open('nb_model.pkl','rb'))
#rf_model=pickle.load(open('rf_model.pkl','rb'))
#svm=pickle.load(open('svm.pkl','rb'))

def classify(num):
    if num == 1:
        return 'Fatal injuries'
    elif num == 2:
        return 'Serious injuries'
    else:
        return 'Slight injuries'

def main():
    st.title("Case Study - Volkswagen AG")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Prediction of accident severity</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Linear Regression','Logistic Regression','Decision Tree', 'Naive Bayes']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    
    
    #input variables

    latitude = st.number_input('Please specify the latitude in the range of [50.10 - 60.15]')
    longitude = st.number_input('Please specify the longitude in the range of [-7.64 - 1.75]')
    vehicles = st.slider("How many vehicles are involved?", 0, 10)
    casualties = st.slider("How many people are involved?", 0, 10)
    speed_limit = st.select_slider('What is the current speed limit? Unit should be miles per hours (mph)', options=[10, 15, 20, 30, 40, 50, 60, 70])
    day_of_Week = st.radio('Which weekday is it?', ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
    road_Class = st.radio('What class of road?', ["Motorway","A(M)","A","B","C","Unclassified"])
    junction_Control = st.radio('Is there someone/something who controls the junction?', ["Not at junction or within 20 metres","Authorised person","Auto traffic signal","Stop sign","Give way or uncontrolled","Data missing or out of range"])
    
    pedestrian_Crossing_Human_Control = st.radio('Are there physical crossing controls?', ["None within 50 metres",
                                                                                           "Control by school crossing patrol",
                                                                                           "Control by other authorised person","Data missing"])
    light_Conditions = st.radio('Is it day or night?', ["Daylight","Darkness - lights lit","Darkness - lights unlit",
                                                        "Darkness- no lighting","Darkness - lighting unknown","Data is missing"])
    urban_or_Rural_Area = st.radio('Is the area urban, rural or unallocated?', ["Urban", "Rural", "Unallocated"])  
    road_Type = st.selectbox('What type of road?', ["Roundabout", "One way street", "Dual carriageway",
                                                         "Single carriageway", "Slip road", "Unknown",
                                                         "One way street/Slip road", "Data is missing"])
    weather_Conditions = st.selectbox('How is the weather?', ["Fine no high winds", "Raining no high winds",
                                                              "Snowing no high winds", "Fine + high winds", 
                                                              "Raining + high winds", "Snowing + high winds",
                                                              "Fog or mist", "Other", "Unknown", "Data missing"])
    road_Surface_Conditions = st.selectbox('How is the road surface conditions?', ["Dry", "Wet or damp", "Snow", "Frost or ice", 
                                                                                   "Flood over 3cm. deep", "Oil or diesel", "Mud",
                                                                                   "Data missing"])
    special_Conditions_at_Site = st.selectbox('Are there any special conditions on the road?', ["None", 
                                                                                                "Auto traffic signal - out",
                                                                                                "Auto signal part defective",
                                                                                                "Road sign or marking defective or obscured",
                                                                                                "Road surface defective", "Oil or diesel",
                                                                                                "Mud", "Data missing"])
    carriageway_Hazards = st.selectbox('Are there obstacles on the road?', ["None","Vehicle load on road",
                                                                            "Other object on road", "Previous accident",
                                                                            "Dog on road",
                                                                            "Other animal on road", 
                                                                            "Pedestrian in carriageway - not injured",
                                                                            "Any animal in carriageway (except ridden horse)",
                                                                            "Data missing"])
    regions = ['East Midlands (England)', 'East of England', 'London',
                                            'North East (England)', 'North West (England)', 'Scotland',
                                            'South East (England)', 'South West (England)', 'Wales','West Midlands (England)',
                                            'Yorkshire and The Humber']
    region = st.selectbox('Which region?', regions)
    
    
    #region
    r1=0
    r2=0
    r3=0
    r4=0
    r5=0
    r6=0
    r7=0
    r8=0
    r9=0
    r10=0
    
    if region == 'East Midlands (England)':
        r1=0
    elif region == 'East of England':
        r1=1
    elif region == 'London':
        r2=1
    elif region == 'North East (England)':
        r3=1
    elif region == 'North West (England)':
        r4=1
    elif region == 'Scotland':
        r5=1
    elif region == 'South East (England)':
        r6=1
    elif region == 'South West (England)':
        r7=1
    elif region == 'Wales':
        r8=1
    elif region == 'West Midlands (England)':
        r9=1
    else:
        r10=1 

    
    
    #normalisation of input based on prior min/max of dataset
    
    longitude = (longitude-accidentsNotNorm["Longitude"].min())/(accidentsNotNorm["Longitude"].max()-accidentsNotNorm["Longitude"].min())
    latitude = (latitude-accidentsNotNorm["Latitude"].min())/(accidentsNotNorm["Latitude"].max()-accidentsNotNorm["Latitude"].min())
    vehicles = (vehicles-accidentsNotNorm["Number_of_Vehicles"].min())/(accidentsNotNorm["Number_of_Vehicles"].max()-accidentsNotNorm["Number_of_Vehicles"].min())
    casualties = (casualties-accidentsNotNorm["Number_of_Casualties"].min())/(accidentsNotNorm["Number_of_Casualties"].max()-accidentsNotNorm["Number_of_Casualties"].min())
    road_Class = (map(road_Class,dict_road_class)-accidentsNotNorm["1st_Road_Class"].min())/(accidentsNotNorm["1st_Road_Class"].max()-accidentsNotNorm["1st_Road_Class"].min())
    road_Type = (map(road_Type,dict_road_type)-accidentsNotNorm["Road_Type"].min())/(accidentsNotNorm["Road_Type"].max()-accidentsNotNorm["Road_Type"].min())
    speed_limit = (speed_limit-accidentsNotNorm["Speed_limit"].min())/(accidentsNotNorm["Speed_limit"].max()-accidentsNotNorm["Speed_limit"].min())
    junction_Control = (map(junction_Control,dict_junction_control)-accidentsNotNorm["Junction_Control"].min())/(accidentsNotNorm["Junction_Control"].max()-accidentsNotNorm["Junction_Control"].min())
    pedestrian_Crossing_Human_Control = (map(pedestrian_Crossing_Human_Control,dict_pd_human_control)-accidentsNotNorm["Pedestrian_Crossing-Human_Control"].min())/(accidentsNotNorm["Pedestrian_Crossing-Human_Control"].max()-accidentsNotNorm["Pedestrian_Crossing-Human_Control"].min())
    light_Conditions = (map(light_Conditions,dict_light_condition)-accidentsNotNorm["Light_Conditions"].min())/(accidentsNotNorm["Light_Conditions"].max()-accidentsNotNorm["Light_Conditions"].min())
    weather_Conditions = (map(weather_Conditions,dict_weather_condition)-accidentsNotNorm["Weather_Conditions"].min())/(accidentsNotNorm["Weather_Conditions"].max()-accidentsNotNorm["Weather_Conditions"].min())
    road_Surface_Conditions = (map(road_Surface_Conditions,dict_road_surface)-accidentsNotNorm["Road_Surface_Conditions"].min())/(accidentsNotNorm["Road_Surface_Conditions"].max()-accidentsNotNorm["Road_Surface_Conditions"].min())
    special_Conditions_at_Site = (map(special_Conditions_at_Site,dict_special_conditions)-accidentsNotNorm["Special_Conditions_at_Site"].min())/(accidentsNotNorm["Special_Conditions_at_Site"].max()-accidentsNotNorm["Special_Conditions_at_Site"].min())
    carriageway_Hazards = (map(carriageway_Hazards,dict_hazards)-accidentsNotNorm["Carriageway_Hazards"].min())/(accidentsNotNorm["Carriageway_Hazards"].max()-accidentsNotNorm["Carriageway_Hazards"].min())
    urban_or_Rural_Area = (map(urban_or_Rural_Area,dict_urban_rural)-accidentsNotNorm["Urban_or_Rural_Area"].min())/(accidentsNotNorm["Urban_or_Rural_Area"].max()-accidentsNotNorm["Urban_or_Rural_Area"].min())
    day_of_Week = (map(day_of_Week,dict_week)-accidentsNotNorm["Day_of_Week"].min())/(accidentsNotNorm["Day_of_Week"].max()-accidentsNotNorm["Day_of_Week"].min())

    
    inputs=[[longitude, latitude, vehicles, casualties, day_of_Week, road_Class, road_Type, speed_limit, junction_Control,
            pedestrian_Crossing_Human_Control, light_Conditions, weather_Conditions, road_Surface_Conditions,
            special_Conditions_at_Site, carriageway_Hazards, urban_or_Rural_Area, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10]]
    
    #classify button
    if st.button('Classify'):
        if option=='Linear Regression':
            st.success(classify(lin_model.predict(inputs)))
        elif option=='Logistic Regression':
            st.success(classify(log_model.predict(inputs)))
        elif option=='Decision Tree':
            st.success(classify(dt.predict(inputs)))
        #elif option=='Random Forest':
         #   st.success(classify(dt.predict(inputs)))
        else:
            st.success(classify(nb_model.predict(inputs)))


# In[ ]:


if __name__=='__main__':
    main()

