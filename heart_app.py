import streamlit as st
import pandas as pd
import numpy as np
import joblib

# PAGE CONFIG 
st.set_page_config(
    page_title="Heart Stroke Predictor",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# LOAD ARTIFACTS 
@st.cache_resource
def load_artifacts():
    return (
        joblib.load("KNN_heart.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("columns.pkl"),
    )

model, scaler, cols = load_artifacts()

# CUSTOM CSS (Mint Palette) 
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp      { background:#000; }
        .stButton>button {
            background:#17A589; color:#fff; border:none; border-radius:12px;
            font-weight:600; padding:.6rem 2rem; transition:all .25s;
        }
        .stButton>button:hover {
            background:#154360; transform:translateY(-2px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# TITLE 
st.title("❤️ Heart Stroke Prediction")

st.markdown("Provide the following details 👇")

# INPUT WIDGETS 
age = st.slider("🎂 Age", 18, 100, 40)
gender = st.selectbox("🧑‍⚕️ Gender", ["M", "F"])
chest_pain = st.selectbox("💢 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("💉 Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("🧬 Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("🩸 Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("📊 Resting ECG", ["NORMAL", "ST", "LVH"])
max_hr = st.slider("❤️‍🔥 Maximum Heart Rate", 60, 220, 130)
exercise_angina = st.selectbox("🏃 Exercise-Induced Angina", ["Yes", "No"])
oldpeak = st.slider("📉 Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
st_slope = st.selectbox("📈 ST Slope", ["UP", "FLAT", "DOWN"])

# PREDICTION 
if st.button("🔍 Predict"):
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "OldPeak": oldpeak,
        "Gender" + gender: 1,
        "ChestPainType" + chest_pain: 1,
        "RestingECG" + resting_ecg: 1,
        "ExerciseAngina" + exercise_angina: 1,
        "STSlope" + st_slope: 1,
    }

    # Ensure all model columns exist
    input_df = pd.DataFrame([raw_input])
    for col in cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[cols]

    # Scale and predict
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]

    # RESULT 
    if pred == 1:
        st.error("🚨🔴 **High Risk of Heart Disease Detected!**")
    else:
        st.success("✅🟢 **Low Risk of Heart Disease** – keep it up!")