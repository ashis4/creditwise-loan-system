import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CreditWise Loan System", page_icon="💼", layout="centered")

# --- LOAD MODEL ---
# Assuming your model is named 'model.pkl'. Change if necessary.
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>💼 CreditWise</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-Powered Loan Approval & Advisory System</p>", unsafe_allow_html=True)
st.write("---")

# --- MAIN UI (USER INPUTS) ---
# Keeping your original layout logic
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=120)
    loan_amount_term = st.number_input("Loan Amount Term", min_value=0, value=360)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# --- PREDICTION LOGIC ---
if st.button("Predict Loan Status", use_container_width=True):
    # Encoding inputs to match your model training
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0
    education_val = 0 if education == "Graduate" else 1
    employed_val = 1 if self_employed == "Yes" else 0
    
    dep_val = 3 if dependents == "3+" else int(dependents)
    
    prop_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    prop_val = prop_map[property_area]

    features = np.array([[gender_val, married_val, dep_val, education_val, employed_val, 
                          applicant_income, coapplicant_income, loan_amount, 
                          loan_amount_term, credit_history, prop_val]])

    prediction = model.predict(features)
    
    if prediction == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")

# --- THE NEW COMPACT PROFESSIONAL FOOTER ---
# This replaces your old footer entirely
st.write("##") # Spacing before footer
st.write("---")

footer_content = """
<style>
/* Hide Streamlit's default footer */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

.footer-container {
    background-color: #161a24; /* Dark theme matching your image */
    color: #e0e0e0;
    padding: 30px 20px 15px 20px;
    font-family: 'Inter', sans-serif;
    border-radius: 10px;
}

.footer-top {
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
}

.footer-section-title {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    color: #888e9b;
    margin-bottom: 12px;
}

.footer-text {
    font-size: 14px;
    line-height: 1.6;
}

.footer-link {
    display: block;
    color: #ffffff;
    text-decoration: none;
    font-size: 14px;
    margin-bottom: 8px;
    transition: 0.3s;
}

.footer-link:hover {
    color: #ff4b4b;
}

.footer-bottom {
    border-top: 1px solid #2d323e;
    padding-top: 15px;
    text-align: center;
    font-size: 12px;
    color: #636a77;
}

.social-icons {
    display: flex;
    gap: 15px;
    margin-top: 10px;
}

/* Responsive adjustment */
@media (max-width: 600px) {
    .footer-top { grid-template-columns: 1fr; text-align: center; }
    .social-icons { justify-content: center; }
}
</style>

<div class="footer-container">
    <div class="footer-top">
        <div>
            <div class="footer-section-title">Developer Credits</div>
            <p class="footer-text">Developed by <b>Ashish Gaikar</b><br>
            Built using Machine Learning with dataset analysis in Jupyter Lab.</p>
        </div>
        
        <div>
            <div class="footer-section-title">Connect</div>
            <a href="https://github.com/ashis4" class="footer-link">🐙 View on GitHub</a>
            <div class="social-icons">
                <a href="#" style="text-decoration:none;">🔵 LinkedIn</a>
                <a href="#" style="text-decoration:none;">📧 Email</a>
            </div>
        </div>

        <div>
            <div class="footer-section-title">About</div>
            <a href="#" class="footer-link">About Ashish</a>
            <a href="#" class="footer-link">Technology Stack</a>
            <a href="#" class="footer-link">FAQ</a>
        </div>

        <div>
            <div class="footer-section-title">Legal</div>
            <a href="#" class="footer-link">Terms of Service</a>
            <a href="#" class="footer-link">Privacy Policy</a>
        </div>
    </div>
    
    <div class="footer-bottom">
        © 2026 CreditWise. All rights reserved.
    </div>
</div>
"""

st.markdown(footer_content, unsafe_allow_html=True)
