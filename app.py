
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from geopy.distance import geodesic

# Load model and features
model = joblib.load("best_fire_risk_model_prob.pkl")
features = joblib.load("model_features_prob.pkl")

st.set_page_config(page_title="🔥 Fire Risk Dashboard", layout="wide")
st.title("🔥 Fire Outbreak Prediction and Hydrant Map Dashboard")

# 1. 📥 User Inputs
st.sidebar.header("Input Parameters")
temperature = st.sidebar.slider("Temperature (°C)", 20, 50, 35)
traffic = st.sidebar.selectbox("Traffic Level", ['Low', 'Moderate', 'High'])
road = st.sidebar.selectbox("Road Condition", ['Good', 'Fair', 'Poor'])
hydrant_status = st.sidebar.selectbox("Hydrant Status", ['Active', 'Damaged', 'Inactive'])
hour = st.sidebar.slider("Report Hour (0-23)", 0, 23, 14)
day_of_week = st.sidebar.selectbox("Day of the Week", list(range(7)))
satellite_score = st.sidebar.slider("Satellite Risk Score", 0.0, 1.0, 0.5, 0.01)

# 2. 📊 Prepare Input for Prediction
input_df = pd.DataFrame({
    'temperature_c': [temperature],
    'traffic_level': [traffic],
    'road_condition': [road],
    'Hydrant_Status_Code': [hydrant_status],
    'report_hour': [hour],
    'day_of_week': [day_of_week],
    'satellite_risk_score': [satellite_score]
})

# Convert categorical to dummies
input_df = pd.get_dummies(input_df)
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[features]

# 3. 🔮 Predict Fire Risk Probability
risk_prob = model.predict_proba(input_df)[0][1]
st.subheader(f"🔥 Predicted Fire Risk Probability: {risk_prob:.2%}")

# 4. 🗺 Load Hydrant and Incident Data
hydrants_df = pd.read_csv("hydrants_with_coords.csv")
incidents_df = pd.read_csv("fire_incidents.csv")

# 5. 🗺 Display Map
st.subheader("📍 Fire Hydrants & Incidents Map")
map_center = [hydrants_df['Hydrant_Latitude'].mean(), hydrants_df['Hydrant_Longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=13)

# Hydrants
hydrant_cluster = MarkerCluster(name="Hydrants").add_to(m)
for i, row in hydrants_df.iterrows():
    folium.Marker(
        [row['Hydrant_Latitude'], row['Hydrant_Longitude']],
        icon=folium.Icon(color='blue', icon='tint', prefix='fa'),
        popup=f"Hydrant ID: {row['Matched_Hydrant_ID']}<br>Status: {row['Hydrant_Status']}"
    ).add_to(hydrant_cluster)

# Incidents
incident_cluster = MarkerCluster(name="Fire Incidents").add_to(m)
for i, row in incidents_df.iterrows():
    folium.Marker(
        [row['latitude'], row['longitude']],
        icon=folium.Icon(color='red', icon='fire', prefix='fa'),
        popup=f"Barangay: {row['barangay']}<br>Damage: ₱{row['estimated_damage']:,.0f}"
    ).add_to(incident_cluster)

st_folium(m, width=1000, height=500)

# 6. 📊 Risk Score Table by Barangay
if 'barangay' in incidents_df.columns:
    grouped = incidents_df.groupby('barangay')['estimated_damage'].mean().reset_index()
    grouped['fire_risk_probability'] = grouped['estimated_damage'].apply(lambda x: min(x / 1000000, 1.0))
    st.subheader("📍 Risk Score by Barangay (Simulated Probabilities)")
    st.dataframe(grouped.sort_values(by='fire_risk_probability', ascending=False).reset_index(drop=True))
