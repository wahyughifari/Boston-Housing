import streamlit as st
import eda
import prediction

st.set_page_config(page_title="Boston House App", layout="centered")

# Sidebar untuk navigasi
page = st.sidebar.selectbox("Select Page", ["EDA", "Predict House Price"])

# Menjalankan halaman sesuai pilihan
if page == "EDA":
    eda.run()
else:
    prediction.run()