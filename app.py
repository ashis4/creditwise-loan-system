import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="CreditWise | Loan Prediction System",
    page_icon="💰",
    layout="centered"
)

# --- CUSTOM CSS FOR FOOTER & STYLING ---
footer_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { bottom: 50px; }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #FAFAFA;
        text-align: center;
        padding: 15px;
        font-size: 14px;
        border-top: 1px solid #31333F;
        z-index: 100;
    }
    .footer a {
        color: #FF4B4B;
        text-decoration: none;
        font-weight: bold;
        margin: 0 10px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
"""
st.markdown(footer_style, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    # Replace with your actual filenames
    model = joblib.load('model.pkl')
    # If you used a scaler during training, load it here
    # scaler = joblib.load('scaler.pkl') 
    return model

model = load_models()

# --- HEADER SECTION ---
st.title("💰 CreditWise Loan System")
st.subheader("Predict Loan Approval in Seconds")
st.write("Fill in the details below to check the eligibility status.")
st.markdown("---")

# --- INPUT FORM ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        
    with col2:
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        credit_history = st.selectbox("Credit History", ["1.0", "0.0"])

    st.markdown("### Financial Details")
    c3, c4, c5 = st.columns(3)
    with c3:
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
    with c4:
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0)
    with c5:
        loan_amount = st.number_input("Loan Amount ($k)", min_value=0, value=120)
    
    loan_term = st.slider("Loan Amount Term (Days)", 12, 480, 360)

# --- PREPROCESSING LOGIC ---
# Mapping inputs to the numerical format your Naive Bayes model expects
def preprocess_inputs():
    # Example mappings (Adjust based on your specific LabelEncoder/Mapping)
    gen = 1 if gender == "Male" else 0
    mar = 1 if married == "Yes" else 0
    dep = 3 if dependents == "3+" else int(dependents)
    edu = 0 if education == "Graduate" else 1
    emp = 1 if self_employed == "Yes" else 0
    prop = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    cred = float(credit_history)
    
    # Combine into a numpy array for prediction
    features = np.array([[gen, mar, dep, edu, emp, applicant_income, 
                          coapplicant_income, loan_amount, loan_term, cred, prop]])
    return features

# --- PREDICTION ---
st.write("##")
if st.button("Check Eligibility", type="primary", use_container_width=True):
    with st.spinner("Analyzing credit profile..."):
        input_data = preprocess_inputs()
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        st.write("##")
        if prediction == 1:
            st.success(f"### 🎉 Congratulations! Loan Likely Approved")
            st.info(f"Confidence Score: {probability:.2%}")
        else:
            st.error(f"### ❌ Loan Application Rejected")
            st.warning(f"Confidence Score: {probability:.2%}")

# --- STICKY FOOTER ---
st.markdown(
    f"""
    <div class="footer">
        Developed by <a href="https://github.com/ashis4" target="_blank">Ashis</a> | 
        <a href="https://github.com/ashis4/creditwise-loan-system" target="_blank">View GitHub Repo</a> |
        Built with Streamlit & Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True
)
