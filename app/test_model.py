import pandas as pd
import joblib

# Load model, encoders, columns
model = joblib.load("../models/model.pkl")
encoders = joblib.load("../models/encoders.pkl")
columns = joblib.load("../models/columns.pkl")

print("=== Testing Model with Sample Data ===\n")

# 🔴 Test Sample 1 (High Churn Risk)
sample1 = {
    'gender': 'Female',
    'SeniorCitizen': 1,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 95,
    'TotalCharges': 95
}

# 🟢 Test Sample 2 (Low Churn Risk)
sample2 = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'Yes',
    'tenure': 60,
    'PhoneService': 'Yes',
    'MultipleLines': 'Yes',
    'InternetService': 'DSL',
    'OnlineSecurity': 'Yes',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'Yes',
    'TechSupport': 'Yes',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Two year',
    'PaperlessBilling': 'No',
    'PaymentMethod': 'Bank transfer (automatic)',  # ✅ correct value
    'MonthlyCharges': 40,
    'TotalCharges': 2500
}

samples = [sample1, sample2]

for i, sample in enumerate(samples, 1):
    print(f"\n🔹 Test Sample {i}")

    df = pd.DataFrame([sample])

    # 🔥 Safe encoding (handles unknown values)
    for col in df.columns:
        if col in encoders:
            le = encoders[col]
            value = df[col].iloc[0]

            if value in le.classes_:
                df[col] = le.transform(df[col])
            else:
                print(f"⚠️ Unknown value '{value}' in column '{col}', using default")
                df[col] = le.transform([le.classes_[0]])

    # Convert numeric columns
    df["tenure"] = df["tenure"].astype(float)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # Predict
    pred = model.predict(df)
    proba = model.predict_proba(df)

    print(f"Stay Probability: {proba[0][0]*100:.2f}%")
    print(f"Churn Probability: {proba[0][1]*100:.2f}%")

    if pred[0] == 1:
        print("Result: 🔴 Customer will CHURN")
    else:
        print("Result: 🟢 Customer will STAY")

print("\n✅ Testing Completed Successfully!")