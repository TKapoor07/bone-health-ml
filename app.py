import streamlit as st
import pandas as pd
import joblib


# Load the trained ML pipeline
model = joblib.load("models/bone_health_model.pkl")


# Page configuration
st.set_page_config(
    page_title="Bone Health Risk Assessment",
    layout="centered"
)


# Title
st.title("Bone Health Risk Assessment")
st.write(
    "Enter the user's information below to estimate "
    "their predicted bone health risk."
)


# -----------------------------
# User inputs
# -----------------------------
st.subheader("Personal Information")
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=55
)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=24.5,
    step=0.1
)
st.subheader("Lifestyle")
physical_activity = st.selectbox(
    "Vigorous physical activity?",
    ["No", "Yes"]
)

smoking = st.selectbox(
    "Smoked at least 100 cigarettes in lifetime?",
    ["No", "Yes"]
)

alcohol_options = {
    "Never": 1,
    "Once a year or less": 2,
    "2–4 times a year": 3,
    "5–9 times a year": 4,
    "10–19 times a year": 5,
    "20–39 times a year": 6,
    "40–99 times a year": 7,
    "100 or more times a year": 8,
    "Daily or almost daily": 9,
    "Other": 10
}

alcohol_label = st.selectbox(
    "Alcohol frequency (past 12 months)",
    list(alcohol_options.keys())
)

alcohol = alcohol_options[alcohol_label]
st.subheader("Medical History")
previous_fracture = st.selectbox(
    "Have you ever had a fracture after age 20?",
    ["No", "Yes"]
)

steroid_use = st.selectbox(
    "Have you used prednisone or cortisone for a long period?",
    ["No", "Yes"]
)

parental_osteoporosis = st.selectbox(
    "Has a parent ever been diagnosed with osteoporosis?",
    ["No", "Yes"]
)


# -----------------------------
# Convert UI values to model codes
# -----------------------------

sex_code = 2 if sex == "Female" else 1

activity_code = 1 if physical_activity == "Yes" else 2

smoking_code = 1 if smoking == "Yes" else 2

fracture_code = 1 if previous_fracture == "Yes" else 2

steroid_code = 1 if steroid_use == "Yes" else 2

parental_code = 1 if parental_osteoporosis == "Yes" else 2


# -----------------------------
# Create model input
# -----------------------------

user_data = pd.DataFrame([{
    "RIDAGEYR": age,
    "RIAGENDR": sex_code,
    "BMXBMI": bmi,
    "PAQ650": activity_code,
    "SMQ020": smoking_code,
    "ALQ121": alcohol,
    "OSQ080": fracture_code,
    "OSQ130": steroid_code,
    "OSQ150": parental_code
}])


# -----------------------------
# Prediction
# -----------------------------

if st.button("Assess Bone Health Risk"):

    prediction = model.predict(user_data)[0]

    probability = model.predict_proba(user_data)[0][1]

    st.divider()
    st.subheader("Assessment Result")

    st.metric(
        "Model Risk Probability",
        f"{probability * 100:.1f}%"
    )

    if prediction == 1:
        st.warning("### 🟠 Higher Predicted Risk")
    else:
        st.success("### 🟢 Lower Predicted Risk")

    st.info(
        "This is an AI-based risk assessment prototype, "
        "not a medical diagnosis."
    )
