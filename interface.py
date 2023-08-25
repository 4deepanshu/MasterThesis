import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import datetime
import googlemaps

# Load the pre-trained model
model = pickle.load(open('model.h5', 'rb'))
models = pickle.load(open('models.h5', 'rb'))

# Set up the Google Maps API client
gmaps = googlemaps.Client(key='AIzaSyAc4-8_ZK6lmnnt_i3qXByD83jeCb2jcgA')

# Define the column names
# column_names = ['behav', 'b_id', 'timestamp', 'temp', 'humid', 'prep', 'pres', 'w_sp']

# Create the Streamlit interface
st.title("Depot Prediction")
st.write("Enter the values below:")
# Calculate the distance between the locations

start_location = st.text_input("Enter Start Location:")
destination_location = st.text_input("Enter Destination Location:")

behav = []
duration = []
if start_location and destination_location:
    directions_result = gmaps.directions(start_location, destination_location, mode="driving", units='metric')
    if directions_result:
        distance_s = directions_result[0]['legs'][0]['distance']['text']
        durations = directions_result[0]['legs'][0]['duration']['text']
        distance = float(distance_s.split()[0])
        duration.append(durations)
        if float(distance) < 10:
            behav.append(0)
        elif 10 <= float(distance) <= 100:
            behav.append(1)
        elif 100 <= float(distance) <= 200:
            behav.append(2)
        else:
            behav.append(3)            

# Input form
input_form = {}

b_id = st.slider("Select b_id", min_value=50230, max_value=51550, step=1)
date = st.date_input("Enter the date")
# Create a time input field for the time
time = st.time_input("Enter the time")
# Combine the date and time into a datetime object
timestamp = datetime.datetime.combine(date, time)
# Convert the timestamp to Unicode
timestamp = str(int(timestamp.timestamp()))

temp = st.slider("Select temperature ", 0.0, 30.0, step=0.1)
humid = st.slider("Humidity", 0.0, 20.0, step=0.1)
prep = st.slider("Precipitation ", 0.00, 6.00, step=0.01)
pres = st.slider("Pressure ", 96.0, 103.0, step=0.1)
w_sp = st.slider("Wind speed ", 0.0, 30.0, step=0.1)

column_names = ['behav', 'b_id', 'timestamp', 'temp', 'humid', 'prep', 'pres', 'w_sp']

# Create an empty DataFrame with the column names
input_data = pd.DataFrame(columns=column_names)
input_data.loc[0, 'behav'] = behav[0]
input_data.loc[0, 'b_id'] = b_id
input_data.loc[0, 'timestamp'] = timestamp
input_data.loc[0, 'temp'] = temp
input_data.loc[0, 'humid'] = humid
input_data.loc[0, 'prep'] = prep
input_data.loc[0, 'pres'] = pres
input_data.loc[0, 'w_sp'] = w_sp

# Convert columns to appropriate data types
input_data['behav'] = input_data['behav'].astype(int)
input_data['b_id'] = input_data['b_id'].astype(int)
input_data['timestamp'] = input_data['timestamp'].astype(int)
input_data['temp'] = input_data['temp'].astype(float)
input_data['humid'] = input_data['humid'].astype(float)
input_data['prep'] = input_data['prep'].astype(float)
input_data['pres'] = input_data['pres'].astype(float)
input_data['w_sp'] = input_data['w_sp'].astype(float)

# Convert input data to DMatrix

dinput = xgb.DMatrix(input_data[column_names])

# Predict department
spot_pred = models.predict(dinput)[0]

input_data['spot'] = spot_pred + 1
dinput = xgb.DMatrix(input_data[column_names])
depot_pred = model.predict(dinput)[0]

depotname = "xx"
depotn = int(depot_pred)
if depotn == 0:
    depotname = "15 Rathausstraße Berlin 10178 DE, 10178 Berlin"
elif depotn == 1:
    depotname = "18 Markgrafenstraße Berlin 10969 DE, 10969 Berlin"
elif depotn == 2:
    depotname = "28 Markgrafenstraße Berlin 10117 DE, 10117 Berlin"
elif depotn == 3:
    depotname = "DE Q-Park, Alexanderstraße, 10178 Berlin"
elif depotn == 4:
    depotname = "Holzmarktstraße 12-14, 10179 Berlin"
elif depotn == 5:
    depotname = "Karl-Liebknecht-Str. 13, 10178 Berlin"
elif depotn == 6:
    depotname = "Karl-Liebknecht-Straße 5, U2 104, 10178 Berlin"
elif depotn == 7:
    depotname = "Seydelstraße 7, 10117 Berlin"
elif depotn == 8:
    depotname = "Unter den Linden 77, 10117 Berlin"
elif depotn == 9:
    depotname = "Veteranenstraße 25, 10119 Berlin"
else:
    depotname = "XX"






# Predefined list of class labels in the same order as your model's classes
class_labels = ["15 Rathausstraße Berlin 10178 DE, 10178 Berlin", 
                "18 Markgrafenstraße Berlin 10969 DE, 10969 Berlin", 
                "28 Markgrafenstraße Berlin 10117 DE, 10117 Berlin", 
                "DE Q-Park, Alexanderstraße, 10178 Berlin", 
                "Holzmarktstraße 12-14, 10179 Berlin", 
                "Karl-Liebknecht-Str. 13, 10178 Berlin", 
                "Karl-Liebknecht-Straße 5, U2 104, 10178 Berlin", 
                "Seydelstraße 7, 10117 Berlin", 
                "Unter den Linden 77, 10117 Berlin", 
                "Veteranenstraße 25, 10119 Berlin"]

# Predict class raw scores for the input data
class_scores = model.predict(dinput, output_margin=True)

# Apply softmax to convert raw scores to probabilities
import numpy as np
class_probabilities = np.exp(class_scores) / np.sum(np.exp(class_scores))

# Create a dictionary to store class probabilities
class_probabilities_dict = {}
for i, label in enumerate(class_labels):
    class_probabilities_dict[label] = class_probabilities[0][i]

# Sort the class probabilities in descending order
sorted_classes = sorted(class_probabilities_dict.keys(), key=lambda k: class_probabilities_dict[k], reverse=True)

# Display the predicted department
st.write('Spot:  ', int(spot_pred))

# Output all classes and their probabilities
for i, label in enumerate(sorted_classes):
    probability = class_probabilities_dict[label]
    if i == 0:
        st.write(f"Rank-{i+1}: {label} (highest prob)")
    else:
        st.write(f"Rank{i+1}: {label} ({probability:.4f})")

d = gmaps.directions(start_location, depotname, mode="driving", units='metric')
st.write("Duration:  ", d[0]['legs'][0]['duration']['text'], " (for highest prob class)")



