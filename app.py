
import streamlit as st
import joblib
import numpy as np
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

lr = joblib.load("linear_regression.pkl")
rf = joblib.load("random_forest.pkl")

# -----------------------------
# Title
# -----------------------------

st.title("🎓 Academic Performance Predictor")

st.write(
    "Enter the student's academic details to predict "
    "the performance level."
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
    # Model Predictions
    # -----------------------------

    pred_lr = lr.predict(X_new)[0]
    pred_rf = rf.predict(X_new)[0]

    # Hybrid prediction
    prediction = (pred_lr + pred_rf) / 2

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
        st.success("👍 GOOD")

    elif result == "Average":
        st.warning("📊 AVERAGE")

    else:
        st.error("⚠️ AT RISK")

    # -----------------------------
    # Explainable AI
    # -----------------------------

    st.divider()

    st.subheader("🔎 Why this result?")

    feature_names = [
        "Internal 1",
        "Internal 2",
        "Assignment",
        "Previous GPA"
    ]

    explainer = shap.TreeExplainer(rf)

    shap_values = explainer.shap_values(X_new)

    contributions = shap_values[0]

    explanation = []

    for feature, contribution in zip(
        feature_names,
        contributions
    ):
        explanation.append(
            (feature, abs(contribution), contribution)
        )

    # Sort by importance
    explanation.sort(
        key=lambda x: x[1],
        reverse=True
    )

    st.write(
        "The following factors had the strongest influence "
        "on this prediction:"
    )

    for feature, importance, contribution in explanation:

        if contribution > 0:
            st.write(
                f"🟢 **{feature}** → Positive influence"
            )

        elif contribution < 0:
            st.write(
                f"🔴 **{feature}** → Negative influence"
            )

        else:
            st.write(
                f"⚪ **{feature}** → Low influence"
            )

    st.caption(
        "Explanation is generated using SHAP-based "
        "feature contributions from the Random Forest model."
    )
