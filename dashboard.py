import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
import folium
from streamlit_folium import st_folium

# Dashboard title
st.title("TfL Bus Delay Prediction Dashboard")

# Simulated bus stops with coordinates
bus_stops = {
    "Trafalgar Square": [51.5080, -0.1281],
    "Oxford Circus": [51.5154, -0.1410],
    "Victoria Station": [51.4952, -0.1439],
    "Liverpool Street": [51.5175, -0.0824]
}

# User selects a bus stop
selected_stop = st.selectbox("Select a bus stop:", list(bus_stops.keys()))

# Show map with selected bus stop
location = bus_stops[selected_stop]
map = folium.Map(location=location, zoom_start=15)
folium.Marker(location=location, popup=selected_stop).add_to(map)
st.subheader("Bus Stop Location")
st_folium(map, width=700, height=400)

# Simulate data based on selected stop
np.random.seed(42)
data = pd.DataFrame({
    'hour': np.random.randint(6, 22, 100),
    'traffic_level': np.random.randint(1, 5, 100),
    'is_raining': np.random.randint(0, 2, 100),
})
data['delay_minutes'] = (
    data['hour'] * 0.1 +
    data['traffic_level'] * 2 +
    data['is_raining'] * 3 +
    np.random.normal(0, 1, 100)
)

# Train model
X = data[['hour', 'traffic_level', 'is_raining']]
y = data['delay_minutes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Show metrics and plots
mse = mean_squared_error(y_test, predictions)
st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")

fig1 = px.scatter(x=y_test, y=predictions,
                  labels={'x': 'Actual Delay', 'y': 'Predicted Delay'},
                  title="Actual vs Predicted Bus Delay")
st.plotly_chart(fig1)

importance = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
fig2 = px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance Analysis")
st.plotly_chart(fig2)
