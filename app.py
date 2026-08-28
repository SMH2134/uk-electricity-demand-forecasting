import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="IoT Sensor Data Stream Simulator", layout="wide")

# Initialize session state for sensor data if not already present.
if "sensor_data" not in st.session_state:
    # Create an empty DataFrame to store streamed sensor data.
    st.session_state.sensor_data = pd.DataFrame(columns=["timestamp", "TSD", "ND", "Temperature"])

# Sidebar for user inputs.
st.sidebar.header("Simulated Sensor Inputs")
# User can tweak these input values to simulate sensor readings.
tsd_value = st.sidebar.slider("Total System Demand (TSD)", min_value=0.0, max_value=10000.0, value=5000.0, step=100.0)
nd_value = st.sidebar.slider("Net Demand (ND)", min_value=0.0, max_value=10000.0, value=4000.0, step=100.0)
temperature = st.sidebar.number_input("Ambient Temperature (°C)", min_value=-20.0, max_value=50.0, value=20.0, step=0.5)

# A button to simulate new sensor data ingestion.
if st.sidebar.button("Stream New Sensor Data"):
    new_data = {
        "timestamp": pd.Timestamp.now(),
        "TSD": tsd_value,
        "ND": nd_value,
        "Temperature": temperature
    }
    # Update sensor_data using pd.concat instead of .append.
    new_row = pd.DataFrame([new_data])
    st.session_state.sensor_data = pd.concat(
        [st.session_state.sensor_data, new_row], ignore_index=True
    )
    st.success("New sensor data streamed.")

# Display the current sensor data.
st.header("Simulated IoT Sensor Data Stream")
st.write("Below is the current list of simulated sensor readings:")
st.dataframe(st.session_state.sensor_data)

# Plotting the sensor data if available.
if not st.session_state.sensor_data.empty:
    sensor_df = st.session_state.sensor_data.copy()
    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
    sensor_df = sensor_df.sort_values("timestamp")
    sensor_df.set_index("timestamp", inplace=True)

    st.subheader("Real-Time Sensor Data Chart (TSD and ND)")
    st.line_chart(sensor_df[["TSD", "ND"]])

    st.subheader("Temperature Over Time")
    st.line_chart(sensor_df["Temperature"])
else:
    st.info("No sensor data streamed yet. Use the sidebar to stream new sensor data.")




# Optional: Auto-refresh simulation
# Uncomment the lines below to have the app auto-update every few seconds.
#
# count = st.experimental_get_query_params().get("count", [0])[0]
# st.experimental_set_query_params(count=int(count)+1)
# time.sleep(2)
# st.experimental_rerun()
