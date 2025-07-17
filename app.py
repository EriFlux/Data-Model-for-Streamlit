import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

# Load model and features
model = joblib.load("best_fire_risk_model.pkl")
features = joblib.load("model_features.pkl")

st.title("🔥 Sta. Cruz Fire Risk Dashboard")
st.markdown("Predict fire outbreak risk, view hydrant maps, and track barangay risk scores.")

# --- Section 1: Fire Risk Prediction ---
st.header("📣 Fire Risk Prediction")
with st.form("prediction_form"):
    temp = st.slider("Temperature (°C)", 20, 50, 34)
    traffic = st.selectbox("Traffic Level", ["Light", "Moderate", "Heavy"])
    road = st.selectbox("Road Condition", ["Dry", "Wet", "Obstructed"])
    hydrant_status = st.selectbox("Nearest Hydrant Status", {
        "Operational": 2,
        "Low Pressure": 1,
        "Non-Operational": 0,
        "Damaged": -1,
        "No Hydrant": -2
    })
    hour = st.slider("Hour of Report", 0, 23, 14)
    day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    submit = st.form_submit_button("Predict")

if submit:
    day_map = {"Monday": 0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
    input_df = pd.DataFrame([{
        'temperature_c': temp,
        'traffic_level': traffic,
        'road_condition': road,
        'Hydrant_Status_Code': hydrant_status,
        'report_hour': hour,
        'day_of_week': day_map[day]
    }])
    input_df = pd.get_dummies(input_df)
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    st.success("🔥 High Risk" if prediction == 1 else "✅ Low Risk")

# --- Section 2: Map of Hydrants and Incidents ---
st.header("🗺 Hydrant and Fire Incident Map")

data = pd.read_csv("merged_fire_incidents_with_hydrants.csv")
m = folium.Map(location=[14.28, 121.42], zoom_start=13)
cluster = MarkerCluster().add_to(m)

for _, row in data.iterrows():
    if pd.notnull(row['Hydrant_Latitude']):
        folium.Marker(
            location=[row['Hydrant_Latitude'], row['Hydrant_Longitude']],
            popup=f"{row['barangay'].title()} - Hydrant: {row['Hydrant_Status']}",
            icon=folium.Icon(color="blue", icon="tint")
        ).add_to(cluster)

folium_static(m)

# --- Section 3: Risk Score Leaderboard ---
st.header("📊 Risk Score by Barangay")
risk_df = data.copy()
risk_df['high_risk_area'] = risk_df['estimated_damage'].apply(lambda x: 1 if x >= 500000 else 0)

risk_summary = risk_df.groupby('barangay').agg({
    'estimated_damage': 'mean',
    'high_risk_area': 'sum',
    'Matched_Hydrant_ID': 'count'
}).rename(columns={
    'estimated_damage': 'avg_damage',
    'high_risk_area': 'fire_count',
    'Matched_Hydrant_ID': 'hydrant_count'
})

risk_summary['risk_score'] = (
    (risk_summary['fire_count'] / risk_summary['fire_count'].max()) * 0.4 +
    (risk_summary['avg_damage'] / risk_summary['avg_damage'].max()) * 0.5 -
    (risk_summary['hydrant_count'] / risk_summary['hydrant_count'].max()) * 0.3
)

risk_summary = risk_summary.sort_values('risk_score', ascending=False).reset_index()
st.dataframe(risk_summary)
