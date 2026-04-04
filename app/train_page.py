import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns


def train_page():

    st.title("📂 Train Model")

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.write("📊 Dataset Preview")
        st.dataframe(data.head())

        if st.button("Train Model"):

            print("\n=== TRAINING STARTED ===")
            st.write("🔹 Loading dataset...")

            # ----------------------
            # Cleaning
            # ----------------------
            print("Cleaning data...")
            st.write("🔹 Cleaning data...")

            data.drop("customerID", axis=1, inplace=True, errors='ignore')
            data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
            data.dropna(inplace=True)

            # ----------------------
            # Encoding
            # ----------------------
            print("Encoding categorical features...")
            st.write("🔹 Encoding categorical features...")

            encoders = {}
            for col in data.columns:
                if data[col].dtype == 'object':
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
                    encoders[col] = le

            # ----------------------
            # Split
            # ----------------------
            print("Splitting features and target...")
            st.write("🔹 Splitting features and target...")

            X = data.drop("Churn", axis=1)
            y = data["Churn"]

            # ----------------------
            # SMOTE
            # ----------------------
            print("Applying SMOTE...")
            st.write("🔹 Applying SMOTE (balancing data)...")

            sm = SMOTE(random_state=42)
            X, y = sm.fit_resample(X, y)

            # ----------------------
            # Train-test split
            # ----------------------
            print("Train-test split...")
            st.write("🔹 Splitting train and test data...")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # ----------------------
            # Training
            # ----------------------
            print("Training Random Forest model...")
            st.write("🔹 Training model (Random Forest)...")

            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)

            # ----------------------
            # Prediction
            # ----------------------
            print("Making predictions...")
            st.write("🔹 Making predictions...")

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            print(f"Accuracy: {acc:.4f}")
            st.success(f"✅ Model Trained Successfully! Accuracy: {acc:.4f}")

            # ----------------------
            # Save model
            # ----------------------
            print("Saving model...")
            st.write("🔹 Saving model...")

            os.makedirs("../models", exist_ok=True)
            joblib.dump(model, "../models/model.pkl")
            joblib.dump(encoders, "../models/encoders.pkl")
            joblib.dump(X.columns.tolist(), "../models/columns.pkl")

            print("=== TRAINING COMPLETED ===\n")
            st.write("✅ Training Completed")

            # ----------------------
            # Graphs (SMALL SIZE)
            # ----------------------

            col1, col2 = st.columns(2)

            # Confusion Matrix
            with col1:
                st.write("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)

                fig, ax = plt.subplots(figsize=(3, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

            # Churn Distribution
            with col2:
                st.write("Churn Distribution")

                fig2, ax2 = plt.subplots(figsize=(3, 3))
                sns.countplot(x="Churn", data=data, ax=ax2)
                st.pyplot(fig2)

            # Accuracy Graph
            st.write("Model Accuracy")

            fig3, ax3 = plt.subplots(figsize=(3, 2))
            ax3.bar(["Accuracy"], [acc])
            ax3.set_ylim(0, 1)
            st.pyplot(fig3)