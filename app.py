import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="CreditWise", layout="centered")
st.title("🏦 CreditWise Loan Approval System")

st.markdown("### Enter Applicant Details")

def to_float(value):
    try:
        return float(value)
    except:
        return 0

# Inputs (clean)
income = st.text_input("Applicant Income (Monthly)")
co_income = st.text_input("Coapplicant Income")
loan_amount = st.text_input("Loan Amount")
loan_term = st.text_input("Loan Term (months)")

credit_score = st.text_input("Credit Score")
savings = st.text_input("Savings")

education = st.selectbox("Education Level", ["Graduate", "Postgraduate", "Undergraduate"])
employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Business"])
marital = st.selectbox("Marital Status", ["Single", "Married"])
loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal"])
property_area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])
gender = st.selectbox("Gender", ["Male", "Female"])
employer_category = st.selectbox("Employer Category", ["Private", "Government"])

# Convert
income = to_float(income)
co_income = to_float(co_income)
loan_amount = to_float(loan_amount)
loan_term = to_float(loan_term)
credit_score = to_float(credit_score)
savings = to_float(savings)

# Load model
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    columns = pickle.load(open('columns.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")

# Predict
if st.button("🔍 Predict Loan Status"):

    try:
        emi = loan_amount / loan_term if loan_term != 0 else 0
        total_income = income + co_income
        dti = emi / total_income if total_income != 0 else 0

        st.info(f"📊 Monthly EMI: {round(emi, 2)}")
        st.info(f"📊 DTI Ratio: {round(dti, 3)}")

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

        education_map = {"Graduate":0,"Postgraduate":1,"Undergraduate":2}
        input_df["Education_Level"] = input_df["Education_Level"].map(education_map)

        input_df = input_df.fillna(0)

        input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"] ** 2
        input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2

        cat_cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]

        encoded = encoder.transform(input_df[cat_cols])
        try:
            encoded = encoded.toarray()
        except:
            pass

        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

        input_df = pd.concat([input_df.drop(columns=cat_cols), encoded_df], axis=1)
        input_df = input_df.reindex(columns=columns, fill_value=0)
        input_df = input_df.astype(float)

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        if prediction[0] == 1:
            st.success(f"✅ Loan Approved (Confidence: {round(probability[0][1]*100,2)}%)")
        else:
            st.error(f"❌ Loan Rejected (Risk: {round(probability[0][0]*100,2)}%)")

            st.write("### ❗ Reasons:")
            if dti > 0.5:
                st.write(f"- High DTI Ratio ({round(dti,2)})")
            if credit_score < 650:
                st.write("- Low Credit Score")
            if income < emi * 5:
                st.write("- Income too low compared to EMI")

            st.write("### 💡 Suggestions:")
            if dti > 0.5:
                st.write("- Increase loan term or reduce loan amount")
            if credit_score < 650:
                st.write("- Improve credit score above 700")
            if income < emi * 5:
                st.write("- Increase income or reduce EMI")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
