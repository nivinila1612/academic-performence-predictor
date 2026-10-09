
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Academic Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    lr = joblib.load("linear_regression.pkl")
    rf = joblib.load("random_forest.pkl")
    return lr, rf

try:
    lr, rf = load_models()
except Exception as e:
    st.error("Model files could not be loaded.")
    st.error(str(e))
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Academic Performance Predictor")

st.write(
    "Enter the student's academic details "
    "to predict the performance level."
)

st.divider()

# -----------------------------
# Input Section
# -----------------------------
st.subheader("Student Details")

internal_1 = st.number_input(
    "Internal 1 (out of 50)",
    min_value=0.0,
    max_value=50.0,
    value=25.0
)

internal_2 = st.number_input(
    "Internal 2 (out of 50)",
    min_value=0.0,
    max_value=50.0,
    value=25.0
)

assignment = st.number_input(
    "Assignment (out of 5)",
    min_value=0.0,
    max_value=5.0,
    value=2.5
)

previous_gpa = st.number_input(
    "Previous GPA (out of 10)",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

st.divider()

# -----------------------------
# Feature Names
# -----------------------------
feature_names = [
    "Internal 1",
    "Internal 2",
    "Assignment",
    "Previous GPA"
]

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Performance", use_container_width=True):

    # Convert inputs to 0-100 scale
    internal_1_scaled = (internal_1 / 50) * 100
    internal_2_scaled = (internal_2 / 50) * 100
    assignment_scaled = (assignment / 5) * 100
    previous_gpa_scaled = (previous_gpa / 10) * 100

    X_new = np.array([[
        internal_1_scaled,
        internal_2_scaled,
        assignment_scaled,
        previous_gpa_scaled
    ]])

    # -----------------------------
    # Hybrid Prediction
    # -----------------------------
    try:
        pred_lr = float(lr.predict(X_new)[0])
        pred_rf = float(rf.predict(X_new)[0])

        prediction = (pred_lr + pred_rf) / 2

    except Exception as e:
        st.error("Prediction failed.")
        st.error(str(e))
        st.stop()

    # -----------------------------
    # Performance Category
    # -----------------------------
    if prediction >= 85:
        result = "Excellent"
    elif prediction >= 70:
        result = "Good"
    elif prediction >= 50:
        result = "Average"
    else:
        result = "At Risk"

    st.divider()
    st.subheader("Your Result")

    if result == "Excellent":
        st.success("🌟 EXCELLENT")
    elif result == "Good":
        st.success("GOOD")
    elif result == "Average":
        st.warning("📊 AVERAGE")
    else:
        st.error("⚠️ AT RISK")

    # -----------------------------
    # Explainable AI - SHAP
    # -----------------------------
    st.divider()
    st.subheader("🔎 Why this result?")

    st.write(
        "These explanations show how each input "
        "influences the hybrid model's numerical prediction."
    )

    try:
        # Use the same two models as the hybrid prediction
        def hybrid_predict(X):
            X = np.asarray(X, dtype=float)
            pred_lr_values = lr.predict(X)
            pred_rf_values = rf.predict(X)
            return (pred_lr_values + pred_rf_values) / 2

        # A representative background dataset is required.
        # Create shap_background.csv from training data.
        background_df = pd.read_csv("shap_background.csv")

        # Ensure the columns match the model input order.
        background_df = background_df[feature_names]

        background = background_df.to_numpy(dtype=float)

        if len(background) == 0:
            st.error("SHAP background dataset is empty.")
            st.stop()

        # KernelExplainer explains the combined hybrid prediction.
        explainer = shap.KernelExplainer(
            hybrid_predict,
            background
        )

        shap_values = explainer.shap_values(
            X_new,
            nsamples=100
        )

        contributions = np.asarray(shap_values).reshape(-1)

        if len(contributions) != len(feature_names):
            st.error(
                "SHAP returned an unexpected number of feature values."
            )
            st.stop()

        explanation = sorted(
            zip(feature_names, contributions),
            key=lambda item: abs(item[1]),
            reverse=True
        )

        st.write("**Factors influencing the prediction:**")

        for feature, contribution in explanation:
            if contribution > 0:
                st.write(
                    f"🟢 **{feature}**: pushes the numerical "
                    "prediction higher than the baseline."
                )
            elif contribution < 0:
                st.write(
                    f"🔴 **{feature}**: pushes the numerical "
                    "prediction lower than the baseline."
                )
            else:
                st.write(
                    f"⚪ **{feature}**: no measurable contribution."
                )

        st.caption(
            "SHAP values explain the hybrid numerical prediction "
            "relative to the selected background dataset. "
            "They do not directly explain the category label."
        )

    except FileNotFoundError:
        st.warning(
            "SHAP explanation needs shap_background.csv. "
            "Create this file from representative training data "
            "using the same four features and scaling as the models."
        )
    except Exception as e:
        st.warning("The prediction was generated, but SHAP failed.")
        st.exception(e)
