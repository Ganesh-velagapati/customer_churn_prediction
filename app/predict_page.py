import streamlit as st
import pandas as pd
import joblib


def predict_page():

    st.title("🧪 Test Customer Churn Prediction")

    # ----------------------
    # Load Model
    # ----------------------
    try:
        model = joblib.load("../models/model.pkl")
        encoders = joblib.load("../models/encoders.pkl")
        columns = joblib.load("../models/columns.pkl")
    except:
        st.error("⚠️ Train the model first!")
        return

    st.write("Enter customer details:")

    # ----------------------
    # Input Form
    # ----------------------
    col1, col2 = st.columns(2)

    data = {}

    # LEFT COLUMN
    with col1:
        data["gender"] = st.selectbox("Gender", ["Male", "Female"])
        data["SeniorCitizen"] = st.selectbox("Senior Citizen", [0, 1])
        data["Partner"] = st.selectbox("Partner", ["Yes", "No"])
        data["Dependents"] = st.selectbox("Dependents", ["Yes", "No"])
        data["tenure"] = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)

        data["PhoneService"] = st.selectbox("Phone Service", ["Yes", "No"])
        data["MultipleLines"] = st.selectbox("Multiple Lines", ["Yes", "No"])
        data["InternetService"] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        data["OnlineSecurity"] = st.selectbox("Online Security", ["Yes", "No"])
        data["OnlineBackup"] = st.selectbox("Online Backup", ["Yes", "No"])

    # RIGHT COLUMN
    with col2:
        data["DeviceProtection"] = st.selectbox("Device Protection", ["Yes", "No"])
        data["TechSupport"] = st.selectbox("Tech Support", ["Yes", "No"])
        data["StreamingTV"] = st.selectbox("Streaming TV", ["Yes", "No"])
        data["StreamingMovies"] = st.selectbox("Streaming Movies", ["Yes", "No"])

        data["Contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        data["PaperlessBilling"] = st.selectbox("Paperless Billing", ["Yes", "No"])

        data["PaymentMethod"] = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        data["MonthlyCharges"] = st.number_input("Monthly Charges", value=50.0)
        data["TotalCharges"] = st.number_input("Total Charges", value=100.0)

    # ----------------------
    # Prediction Button
    # ----------------------
    if st.button("Predict"):

        df = pd.DataFrame([data])

        # ----------------------
        # Encoding
        # ----------------------
        for col in df.columns:
            if col in encoders:
                le = encoders[col]
                value = df[col].iloc[0]

                if value in le.classes_:
                    df[col] = le.transform(df[col])
                else:
                    st.warning(f"Unknown value in {col}, using default")
                    df[col] = le.transform([le.classes_[0]])

        # Convert numeric
        df["tenure"] = df["tenure"].astype(float)
        df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
        df["TotalCharges"] = df["TotalCharges"].astype(float)

        # ----------------------
        # Prediction
        # ----------------------
        pred = model.predict(df)
        proba = model.predict_proba(df)

        # ----------------------
        # Output
        # ----------------------
        st.subheader("📊 Prediction Result")

        colA, colB = st.columns(2)

        with colA:
            st.metric("Stay Probability", f"{proba[0][0]*100:.2f}%")

        with colB:
            st.metric("Churn Probability", f"{proba[0][1]*100:.2f}%")

        if pred[0] == 1:
            st.error("🔴 Customer will CHURN")
        else:
            st.success("🟢 Customer will STAY")

        # ----------------------
        # DYNAMIC LOCAL EXPLANATION
        # ----------------------
        st.subheader("🧠 Why this prediction?")

        importances = model.feature_importances_
        feature_importance = dict(zip(columns, importances))

        top_features = sorted(feature_importance, key=feature_importance.get, reverse=True)[:5]

        reasons = []

        # CHURN CASE
        if pred[0] == 1:

            if "tenure" in top_features and data["tenure"] < 12:
                reasons.append("Low tenure contributed to churn")

            if "MonthlyCharges" in top_features and data["MonthlyCharges"] > 80:
                reasons.append("High monthly charges contributed to churn")

            if "Contract" in top_features and data["Contract"] == "Month-to-month":
                reasons.append("Month-to-month contract increased churn risk")

            if "TechSupport" in top_features and data["TechSupport"] == "No":
                reasons.append("Lack of tech support influenced churn")

            if len(reasons) == 0:
                reasons.append("Model detected churn based on combined feature patterns")

        # STAY CASE
        else:

            if "tenure" in top_features and data["tenure"] >= 12:
                reasons.append("Higher tenure supports customer retention")

            if "Contract" in top_features and data["Contract"] != "Month-to-month":
                reasons.append("Long-term contract reduces churn risk")

            if "MonthlyCharges" in top_features and data["MonthlyCharges"] < 80:
                reasons.append("Moderate charges help retain customers")

            if len(reasons) == 0:
                reasons.append("Customer profile indicates stable behavior")

        # Show reasons
        for r in reasons:
            st.write(f"• {r}")
