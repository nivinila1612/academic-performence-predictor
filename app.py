
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

    # Model input
    X_new = np.array([[
        internal_1_scaled,
        internal_2_scaled,
        assignment_scaled,
        previous_gpa_scaled
    ]], dtype=float)

    # -----------------------------
    # Hybrid Prediction
    # -----------------------------
    try:
        pred_lr = float(lr.predict(X_new)[0])
        pred_rf = float(rf.predict(X_new)[0])

        prediction = (pred_lr + pred_rf) / 2

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
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

    # -----------------------------
    # Display Result
    # -----------------------------
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
        # Explain the same hybrid model used for prediction
        def hybrid_predict(X):
            X = np.asarray(X, dtype=float)

            pred_lr_values = lr.predict(X)
            pred_rf_values = rf.predict(X)

            return (pred_lr_values + pred_rf_values) / 2

        # Load SHAP background dataset
        background_df = pd.read_csv("shap_background.csv")

        # Fix column-name mismatch
        background_df = background_df.rename(columns={
            "Internal_1": "Internal 1",
            "Internal_2": "Internal 2",
            "Assignment": "Assignment",
            "Previous_GPA": "Previous GPA"
        })

        # Check required columns
        missing_features = [
            name for name in feature_names
            if name not in background_df.columns
        ]

        if missing_features:
            st.error(
                "SHAP background CSV is missing these columns: "
                + ", ".join(missing_features)
            )
            st.stop()

        # Select correct feature order and convert to numbers
        background_df = background_df[feature_names].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        background = background_df.to_numpy(dtype=float)

        if len(background) == 0:
            st.error("SHAP background dataset is empty.")
            st.stop()

        # Limit background rows to improve performance
        if len(background) > 25:
            background = shap.sample(
                background,
                25,
                random_state=42
            )

        # Explain hybrid prediction using KernelExplainer
        explainer = shap.KernelExplainer(
            hybrid_predict,
            background
        )

        shap_values = explainer.shap_values(
            X_new,
            nsamples=100
        )

        # Handle different SHAP output shapes
        contributions = np.asarray(shap_values).reshape(-1)

        if len(contributions) != len(feature_names):
            st.error(
                "SHAP returned an unexpected number of feature values."
            )
            st.stop()

        # Baseline prediction for the background dataset
        baseline = float(
            np.asarray(hybrid_predict(background)).mean()
        )

        st.write(f"**Baseline prediction:** {baseline:.2f}")
        st.write(f"**Hybrid prediction:** {prediction:.2f}")

        # Sort features by contribution magnitude
        explanation = sorted(
            zip(feature_names, contributions),
            key=lambda item: abs(item[1]),
            reverse=True
        )

        st.write("**Factors influencing the prediction:**")

        for feature, contribution in explanation:
            if contribution > 0:
                st.write(
                    f"🟢 **{feature}: +{contribution:.2f}** — "
                    "pushes the numerical prediction higher."
                )
            elif contribution < 0:
                st.write(
                    f"🔴 **{feature}: {contribution:.2f}** — "
                    "pushes the numerical prediction lower."
                )
            else:
                st.write(
                    f"⚪ **{feature}: 0.00** — "
                    "no measurable contribution."
                )

        # SHAP contribution chart
        chart_df = pd.DataFrame(
            explanation,
            columns=["Feature", "SHAP Value"]
        )

        st.write("**SHAP contribution chart**")

        st.bar_chart(
            chart_df.set_index("Feature")["SHAP Value"]
        )

        st.caption(
            "SHAP values explain the hybrid numerical prediction "
            "relative to the selected background dataset. "
            "Positive values push the prediction higher; negative "
            "values push it lower. They do not directly explain "
            "the category label."
        )

    except FileNotFoundError:
        st.warning(
            "SHAP explanation needs shap_background.csv. "
            "Upload this file to the same GitHub repository "
            "as app.py."
        )

    except Exception as e:
        st.warning(
            "The prediction was generated, but SHAP failed."
        )
        st.exception(e)
