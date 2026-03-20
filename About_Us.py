import streamlit as st

st.set_page_config(page_title="About CreditWise", layout="wide")

def load_html(file_name):
    with open(file_name, encoding="utf-8") as f:
        return f.read()

st.markdown(load_html("about.html"), unsafe_allow_html=True)