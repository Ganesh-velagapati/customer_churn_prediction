import streamlit as st

# Dummy admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

def login():

    st.title("🔐 Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state["logged_in"] = True
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")