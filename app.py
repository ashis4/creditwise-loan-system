import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Title
st.title("🏦 CreditWise Loan Approval System")

# Load model files
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    columns = pickle.load(open('columns.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")

# ---------------- INPUTS ---------------- #

st.header("Enter Applicant Details")

income = st.number_input("Applicant Income (Monthly)", min_value=0.0)
co_income = st.number_input("Coapplicant Income (Monthly)", value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_term = st.number_input("Loan Term (in months)", value=60)

credit_score = st.number_input("Credit Score", min_value=0.0)
savings = st.number_input("Savings", value=0.0)

education = st.selectbox("Education Level", ["Graduate", "Postgraduate", "Undergraduate"])
employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Business"])
marital = st.selectbox("Marital Status", ["Single", "Married"])
loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal"])
property_area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])
gender = st.selectbox("Gender", ["Male", "Female"])
employer_category = st.selectbox("Employer Category", ["Private", "Government"])

# ---------------- PREDICTION ---------------- #

if st.button("Predict Loan Status"):

    try:
        # 🔥 EMI Calculation
        if loan_term == 0:
            emi = 0
        else:
            emi = loan_amount / loan_term

        # Total income
        total_income = income + co_income

        # 🔥 Improved DTI (EMI-based)
        if total_income == 0:
            dti = 0
        else:
            dti = emi / total_income

        st.info(f"📊 Monthly EMI: {round(emi, 2)}")
        st.info(f"📊 DTI Ratio: {round(dti, 3)}")

        # Create dataframe
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

        # Convert Education manually
        education_map = {
            "Graduate": 0,
            "Postgraduate": 1,
            "Undergraduate": 2
        }
        input_df["Education_Level"] = input_df["Education_Level"].map(education_map)

        # Handle missing values
        input_df = input_df.fillna(0)

        # Feature engineering
        input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"] ** 2
        input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2

        # Encode categorical variables
        cat_cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]

        encoded = encoder.transform(input_df[cat_cols])

        try:
            encoded = encoded.toarray()
        except:
            pass

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(cat_cols)
        )

        # Combine data
        input_df = pd.concat([input_df.drop(columns=cat_cols), encoded_df], axis=1)

        # Match training columns
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Ensure numeric
        input_df = input_df.astype(float)

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        # Output
        if prediction[0] == 1:
            st.success(f"✅ Loan Approved (Confidence: {round(probability[0][1]*100,2)}%)")
        else:
            st.error(f"❌ Loan Rejected (Risk: {round(probability[0][0]*100,2)}%)")

            # 🔥 Smart Explanation
            st.write("### Reasons:")
            if dti > 0.5:
                st.write(f"- High DTI Ratio ({round(dti,2)})")
            if credit_score < 650:
                st.write("- Low Credit Score")
            if income < emi * 5:
                st.write("- Income is low compared to EMI")

            # 🔥 Suggestions
            st.write("### Suggestions:")
            if dti > 0.5:
                st.write("- Increase loan term or reduce loan amount")
            if credit_score < 650:
                st.write("- Improve credit score above 700")
            if income < emi * 5:
                st.write("- Increase income or reduce EMI burden")

    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")
