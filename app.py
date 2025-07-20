
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
st.title("🔥 Fire Outbreak Prediction and Risk Mapping")

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

# 4. 📄 Load Merged Dataset
df = pd.read_csv("merged_fire_incidents_with_hydrants.csv")

# 5. 🧠 Predict risk for each incident in the dataset (Safe version)

# ✅ Show available columns for debugging (optional)
st.write("📋 Columns in dataset:", df.columns.tolist())

# ✅ Check if required features are present
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    st.error(f"🚫 The following required columns are missing in your dataset: {missing_cols}")
    st.stop()

# ✅ Continue prediction safely
model_input = pd.get_dummies(df[features])
for col in features:
    if col not in model_input.columns:
        model_input[col] = 0
model_input = model_input[features]

df['predicted_risk'] = model.predict_proba(model_input)[:, 1]

# 6. 🗺 Display Map
st.subheader("📍 Fire Incidents & Hydrant Map (from merged file)")
map_center = [df['Hydrant_Latitude'].mean(), df['Hydrant_Longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=13)

cluster = MarkerCluster(name="Fire Points").add_to(m)
for _, row in df.iterrows():
    folium.Marker(
        [row['Hydrant_Latitude'], row['Hydrant_Longitude']],
        icon=folium.Icon(color='red', icon='fire', prefix='fa'),
        popup=f"Barangay: {row['barangay']}<br>Risk: {row['predicted_risk']:.2%}<br>Damage: ₱{row['estimated_damage']:,}"
    ).add_to(cluster)

st_folium(m, width=1000, height=500)

# 7. 📊 Risk Score Table by Barangay (based on model prediction)
st.subheader("📍 Predicted Risk Score by Barangay")
if 'barangay' in df.columns:
    brgy_risk = df.groupby('barangay')['predicted_risk'].mean().reset_index()
    brgy_risk = brgy_risk.sort_values(by='predicted_risk', ascending=False)
    st.dataframe(brgy_risk.reset_index(drop=True))
