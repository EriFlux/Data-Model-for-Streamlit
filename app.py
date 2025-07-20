
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


# 3. 🔘 Predict Fire Risk Probability on Button Click
if st.sidebar.button("🔍 Predict Fire Risk"):
    risk_prob = model.predict_proba(input_df)[0][1]
    st.subheader(f"🔥 Predicted Fire Risk Probability: {risk_prob:.2%}")
else:
    st.subheader("🔥 Adjust parameters on the left, then click 'Predict Fire Risk' to see results.")

risk_prob = model.predict_proba(input_df)[0][1]
st.subheader(f"🔥 Predicted Fire Risk Probability: {risk_prob:.2%}")

# 4. 📄 Load Dataset
df = pd.read_csv("merged_fire_incidents_with_hydrants.csv")

# Try to generate missing model features
if 'date_time_reported' in df.columns:
    df['report_datetime'] = pd.to_datetime(df['date_time_reported'], errors='coerce')
    df['report_hour'] = df['report_datetime'].dt.hour
    df['day_of_week'] = df['report_datetime'].dt.dayofweek

if 'Hydrant_Status' in df.columns:
    df['Hydrant_Status_Code'] = df['Hydrant_Status'].astype('category').cat.codes

if 'satellite_risk_score' not in df.columns:
    df['satellite_risk_score'] = np.random.uniform(0, 1, size=len(df))

df['traffic_level'] = df.get('traffic_level', 'Moderate')
df['road_condition'] = df.get('road_condition', 'Good')

# 🧠 Predict for Dataset
model_input = pd.get_dummies(df)
for col in features:
    if col not in model_input.columns:
        model_input[col] = 0
model_input = model_input[features]
df['predicted_risk'] = model.predict_proba(model_input)[:, 1]

# 5. 📍 Nearest Hydrant Logic
hydrant_locs = df[['Hydrant_Latitude', 'Hydrant_Longitude']].dropna().drop_duplicates().values.tolist()

def get_nearest_hydrant(lat, lon, hydrants):
    try:
        return min(
            ((h_lat, h_lon, geodesic((lat, lon), (h_lat, h_lon)).meters)
             for h_lat, h_lon in hydrants),
            key=lambda x: x[2]
        )
    except:
        return (None, None, None)

df['Nearest_Hydrant_Dist_M'] = df.apply(lambda row: (
    get_nearest_hydrant(row['Hydrant_Latitude'], row['Hydrant_Longitude'], hydrant_locs)[2]
    if pd.notnull(row['Hydrant_Latitude']) and pd.notnull(row['Hydrant_Longitude'])
    else None
), axis=1)

# 6. 🗺 Map with Satellite Option
st.subheader("🗺 Fire & Hydrant Map")
map_style = st.selectbox("🛰 Select Map Tile", ['OpenStreetMap', 'Stamen Terrain', 'Stamen Toner', 'CartoDB Positron', 'CartoDB Dark_Matter'])

map_df = df.dropna(subset=['Hydrant_Latitude', 'Hydrant_Longitude'])
map_center = [map_df['Hydrant_Latitude'].mean(), map_df['Hydrant_Longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=13, tiles=map_style)

cluster = MarkerCluster().add_to(m)
for _, row in map_df.iterrows():
    popup_content = f"Barangay: {row['barangay']}<br>Risk: {row['predicted_risk']:.2%}<br>Damage: ₱{row['estimated_damage']:,}<br>Nearest Hydrant: {row['Nearest_Hydrant_Dist_M']:.2f} m" if pd.notnull(row['Nearest_Hydrant_Dist_M']) else "No hydrant location"
    folium.Marker(
        [row['Hydrant_Latitude'], row['Hydrant_Longitude']],
        icon=folium.Icon(color='red', icon='fire', prefix='fa'),
        popup=popup_content
    ).add_to(cluster)

st_folium(m, width=1000, height=500)

# 7. 📊 Risk Table
st.subheader("📊 Risk Score by Barangay")
if 'barangay' in df.columns:
    summary = df.groupby('barangay').agg({
        'predicted_risk': 'mean',
        'estimated_damage': 'max',
        'Nearest_Hydrant_Dist_M': 'mean',
        'barangay': 'count'
    }).rename(columns={'barangay': 'Incident_Count'}).reset_index()
    summary['predicted_risk'] = summary['predicted_risk'].round(4)
    summary['Nearest_Hydrant_Dist_M'] = summary['Nearest_Hydrant_Dist_M'].round(2)
    summary = summary.sort_values(by='predicted_risk', ascending=False)
    st.dataframe(summary.rename(columns={
        'barangay': 'Barangay',
        'predicted_risk': 'Avg Risk Probability',
        'estimated_damage': 'Max Damage',
        'Nearest_Hydrant_Dist_M': 'Avg Distance to Hydrant (m)'
    }))
