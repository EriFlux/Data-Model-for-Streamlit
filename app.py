
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

# 5. 🧠 Predict risk for each incident in the dataset (Safe + Smart Fixes)

# Try to generate missing model features
if 'date_time_reported' in df.columns:
    df['report_datetime'] = pd.to_datetime(df['date_time_reported'], errors='coerce')
    df['report_hour'] = df['report_datetime'].dt.hour
    df['day_of_week'] = df['report_datetime'].dt.dayofweek

if 'Hydrant_Status' in df.columns:
    df['Hydrant_Status_Code'] = df['Hydrant_Status'].astype('category').cat.codes

# Simulate satellite risk score if not present
if 'satellite_risk_score' not in df.columns:
    df['satellite_risk_score'] = np.random.uniform(0, 1, size=len(df))

# Fill missing traffic/road condition with defaults
df['traffic_level'] = df.get('traffic_level', 'Moderate')
df['road_condition'] = df.get('road_condition', 'Good')

# Create dummy model input
model_input = df.copy()
model_input = pd.get_dummies(model_input)

# Align input columns with training features
for col in features:
    if col not in model_input.columns:
        model_input[col] = 0
model_input = model_input[features]

# Predict fire risk
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
st.subheader("📍 Fire Risk Summary by Barangay (Predicted)")

if 'barangay' in df.columns:
    brgy_summary = df.groupby('barangay').agg({
        'predicted_risk': 'mean',
        'estimated_damage': 'max',
        'barangay': 'count'
    }).rename(columns={'barangay': 'incident_count'}).reset_index()

    brgy_summary['predicted_risk'] = brgy_summary['predicted_risk'].round(4)
    brgy_summary = brgy_summary.sort_values(by='predicted_risk', ascending=False)

    st.dataframe(brgy_summary.rename(columns={
        'barangay': 'Barangay',
        'predicted_risk': 'Avg Risk Probability',
        'estimated_damage': 'Max Damage (₱)',
        'incident_count': 'Incident Count'
    }))
else:
    st.warning("⚠️ 'barangay' column not found in data.")
