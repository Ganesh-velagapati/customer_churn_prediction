import streamlit as st
from login import login

# Page config
st.set_page_config(page_title="Admin Dashboard", layout="wide")

# Session state initialization
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# 🔐 If NOT logged in → show login page
if not st.session_state["logged_in"]:
    login()
    st.stop()

# =========================
# ✅ AFTER LOGIN → DASHBOARD
# =========================

st.sidebar.title("📊 Admin Panel")

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Train Model", "Test Prediction", "Explain Model", "Logout"]
)

# -------------------------
# 🏠 HOME PAGE
# -------------------------
if menu == "Home":
    st.title("🏠 Admin Dashboard")

    st.write("Welcome to Customer Churn Prediction System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Status", "Ready")

    with col2:
        st.metric("Accuracy", "Check in Train Tab")

    with col3:
        st.metric("Version", "1.0")

# -------------------------
# 📂 TRAIN PAGE
# -------------------------
elif menu == "Train Model":
    from train_page import train_page
    train_page()

# -------------------------
# 🧪 TEST PAGE
# -------------------------
elif menu == "Test Prediction":
    from predict_page import predict_page
    predict_page()

# -------------------------
# 🧠 EXPLAIN PAGE
# -------------------------
elif menu == "Explain Model":
    from explain_page import explain_page
    explain_page()

# -------------------------
# 🚪 LOGOUT
# -------------------------
elif menu == "Logout":
    st.session_state["logged_in"] = False
    st.success("Logged out successfully!")
    st.rerun()