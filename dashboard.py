import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px

# Dashboard title
st.title("TfL Bus Delay Prediction Dashboard")

# Simulated list of bus stops (in real case, fetch from TfL API)
bus_stops = {
    "Trafalgar Square": "490008660N",
    "Oxford Circus": "490000254W",
    "Victoria Station": "490000091A",
    "Liverpool Street": "490000235Z"
}

# User selects a bus stop
selected_stop = st.selectbox("Select a Bus Stop:", list(bus_stops.keys()))

# Simulate fetching data for selected stop (in real case, use TfL API)
st.subheader(f"Delay Prediction for: {selected_stop}")
np.random.seed(hash(selected_stop) % 123456)  # simulate different data per stop

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

# Features and target
X = data[['hour', 'traffic_level', 'is_raining']]
y = data['delay_minutes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, predictions)
st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")

# Plot Actual vs Predicted
fig1 = px.scatter(x=y_test, y=predictions,
                  labels={'x': 'Actual Delay', 'y': 'Predicted Delay'},
                  title="Actual vs Predicted Bus Delay")
st.plotly_chart(fig1)

# Feature Importance
importance = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
fig2 = px.bar(importance_df, x='Feature', y='Importance', title="Feature Importance Analysis")
st.plotly_chart(fig2)
