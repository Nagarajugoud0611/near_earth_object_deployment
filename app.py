import streamlit as st
import pickle
import pandas as pd

from neo_features import RAW_FEATURES  # also makes the pipeline's feature function importable

# Load model
with open(r"model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(
    page_title="Asteroid Hazard Prediction",
    page_icon="☄️",
    layout="centered"
)

st.title("☄️ Asteroid Hazard Prediction App")

st.write("Enter the asteroid measurements below:")

# Inputs
est_diameter_min = st.number_input(
    "Estimated Diameter Min (km)",
    min_value=0.0,
    value=0.1,
    format="%.5f"
)

est_diameter_max = st.number_input(
    "Estimated Diameter Max (km)",
    min_value=0.0,
    value=0.2,
    format="%.5f"
)

relative_velocity = st.number_input(
    "Relative Velocity (km/h)",
    min_value=0.0,
    value=48000.0
)

miss_distance = st.number_input(
    "Miss Distance (km)",
    min_value=1.0,
    value=30000000.0
)

absolute_magnitude = st.number_input(
    "Absolute Magnitude (H)",
    value=23.0
)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "est_diameter_min": [est_diameter_min],
        "est_diameter_max": [est_diameter_max],
        "relative_velocity": [relative_velocity],
        "miss_distance": [miss_distance],
        "absolute_magnitude": [absolute_magnitude]
    })[RAW_FEATURES]

    prediction = model.predict(input_data)[0]

    classes = {
        0: "Not Hazardous",
        1: "Hazardous"
    }

    if prediction == 1:
        st.error(f"Prediction: {classes[prediction]} ⚠️")
    else:
        st.success(f"Prediction: {classes[prediction]}")
