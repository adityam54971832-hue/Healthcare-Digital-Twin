
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Healthcare Digital Twin",
    page_icon="🩺",
    layout="wide"
)

st.title("Healthcare Digital Twin")
st.subheader(
    "Explainable AI for Diabetes Outcome Prediction"
)

st.info(
    "Educational demonstration only. "
    "Not intended for diagnosis or medical decisions."
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.joblib")

model = load_model()

# Load sample dataset for SHAP background
data = pd.read_csv("data/diabetes.csv")

X = data.drop("Outcome", axis=1)

# Select a small background sample
background = X.sample(
    n=min(50, len(X)),
    random_state=42
)

# Apply the model's preprocessing
imputer = model.named_steps["imputer"]
scaler = model.named_steps["scaler"]
classifier = model.named_steps["classifier"]

background_imputed = imputer.transform(background)
background_scaled = scaler.transform(background_imputed)

# SHAP explainer for Logistic Regression
explainer = shap.LinearExplainer(
    classifier,
    background_scaled
)

st.header("Virtual Patient Profile")

# User inputs
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0
    )

    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

# Create virtual patient data
patient = pd.DataFrame([{
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": pedigree,
    "Age": age
}])

if st.button("Generate Digital Twin Prediction"):

    # Prediction
    prediction = model.predict(patient)[0]

    probability = model.predict_proba(patient)[0][1]

    st.header("Prediction Result")

    if prediction == 1:
        st.warning(
            "The model predicts dataset Outcome = 1"
        )
    else:
        st.success(
            "The model predicts dataset Outcome = 0"
        )

    st.metric(
        "Model probability for Outcome = 1",
        f"{probability * 100:.2f}%"
    )

    st.caption(
        "This probability is a model output, not a "
        "validated personal health risk."
    )

    # Explain prediction using SHAP
    patient_imputed = imputer.transform(patient)
    patient_scaled = scaler.transform(patient_imputed)

    shap_values = explainer(patient_scaled)

    st.header("Explainable AI - SHAP")

    st.write(
        "The chart shows how each feature contributes "
        "to the model's output."
    )

    values = shap_values.values[0]

    feature_names = patient.columns

    # Feature contribution bar chart
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        feature_names,
        values
    )

    ax.set_xlabel(
        "SHAP contribution to model output (log-odds)"
    )

    ax.set_title(
        "Feature Contributions for This Prediction"
    )

    ax.axvline(0)

    plt.tight_layout()

    st.pyplot(fig)

    st.write(
        "Positive values move the model output toward "
        "Outcome = 1. Negative values move it toward "
        "Outcome = 0. This explains the model, not "
        "the patient's actual medical condition."
    )

    st.header("Digital Twin Summary")

    st.dataframe(patient)


