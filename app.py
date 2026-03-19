import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("🏦 CreditWise Loan Approval System")

# Load files
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

# Inputs
income = st.number_input("Applicant Income")
co_income = st.number_input("Coapplicant Income", value=0)
loan_amount = st.number_input("Loan Amount")
credit_score = st.number_input("Credit Score")
dti = st.number_input("DTI Ratio")
savings = st.number_input("Savings", value=0)

employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Business"])
marital = st.selectbox("Marital Status", ["Single", "Married"])
loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal"])
property_area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])
gender = st.selectbox("Gender", ["Male", "Female"])
employer_category = st.selectbox("Employer Category", ["Private", "Government"])
education = st.selectbox("Education Level", ["Graduate", "Postgraduate", "Undergraduate"])

if st.button("Predict"):

    input_df = pd.DataFrame({
        "Applicant_Income":[income],
        "Coapplicant_Income":[co_income],
        "Loan_Amount":[loan_amount],
        "Credit_Score":[credit_score],
        "DTI_Ratio":[dti],
        "Savings":[savings],
        "Education_Level":[education],
        "Employment_Status":[employment],
        "Marital_Status":[marital],
        "Loan_Purpose":[loan_purpose],
        "Property_Area":[property_area],
        "Gender":[gender],
        "Employer_Category":[employer_category]
    })

    # Feature engineering
    input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"] ** 2
    input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2

    # Encoding
    cat_cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
    encoded = encoder.transform(input_df[cat_cols])

    try:
        encoded = encoded.toarray()
    except:
        pass

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    input_df = pd.concat([input_df.drop(columns=cat_cols), encoded_df], axis=1)

    # Align columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")