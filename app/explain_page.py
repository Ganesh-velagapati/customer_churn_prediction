import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def explain_page():

    st.title("🧠 Model Explainability")

    try:
        model = joblib.load("../models/model.pkl")
        columns = joblib.load("../models/columns.pkl")
    except:
        st.error("⚠️ Train the model first!")
        return

    st.write("### 📊 Feature Importance")

    # Get feature importance
    importances = model.feature_importances_

    df = pd.DataFrame({
        "Feature": columns,
        "Importance": importances
    })

    # Sort top features
    df = df.sort_values(by="Importance", ascending=False).head(8)

    # Plot small graph
    fig, ax = plt.subplots(figsize=(4,3))
    ax.barh(df["Feature"], df["Importance"])
    ax.invert_yaxis()

    st.pyplot(fig)

    # ----------------------
    # Simple Explanation
    # ----------------------
    st.write("### 📌 Interpretation")

    top_features = df.head(3)["Feature"].tolist()

    explanation = f"""
    The model prediction is mainly influenced by:
    - {top_features[0]}
    - {top_features[1]}
    - {top_features[2]}

    Higher impact features contribute more to churn prediction.
    """

    st.info(explanation)