import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

def train_model():
    print("\n🔹 Loading dataset...")
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(os.path.join(base_path, "data", "churn.csv"))

    print("🔹 Cleaning data...")
    data.drop("customerID", axis=1, inplace=True)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
    data.dropna(inplace=True)

    print("🔹 Encoding categorical features...")
    encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le

    print("🔹 Splitting features and target...")
    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    print("🔹 Applying SMOTE (balancing data)...")
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)

    print("🔹 Splitting train and test data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("🔹 Training model (Random Forest)...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    print("🔹 Making predictions...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n✅ Training Completed!")
    print(f"📊 Accuracy: {acc:.4f}")

    print("🔹 Saving model and files...")
    os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
    joblib.dump(model, os.path.join(base_path, "models", "model.pkl"))
    joblib.dump(encoders, os.path.join(base_path, "models", "encoders.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(base_path, "models", "columns.pkl"))
    print("✅ Model saved successfully!\n")

    return acc


# Run directly from CMD
if __name__ == "__main__":
    train_model()
