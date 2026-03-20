import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="CreditWise", layout="wide")

# ---------- LOAD CSS ----------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------- LOAD HEADER ----------
def load_html(file_name):
    with open(file_name, encoding="utf-8") as f:
        return f.read()

st.markdown(load_html("header.html"), unsafe_allow_html=True)

# spacing after header
st.markdown("<div style='margin-top:15px'></div>", unsafe_allow_html=True)

# ---------- SAFE FLOAT ----------
def to_float(value):
    try:
        return float(value)
    except:
        return 0

# ---------- INPUT UI ----------
st.subheader("👤 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    income = st.text_input("Applicant Income (Monthly)")
    credit_score = st.text_input("Credit Score")
    education = st.selectbox("Education Level", ["Graduate", "Postgraduate", "Undergraduate"])
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    co_income = st.text_input("Coapplicant Income")
    savings = st.text_input("Savings")
    employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Business"])
    marital = st.selectbox("Marital Status", ["Single", "Married"])

st.markdown("---")

st.subheader("🏦 Loan Details")

col3, col4 = st.columns(2)

with col3:
    loan_amount = st.text_input("Loan Amount")
    loan_term = st.text_input("Loan Term (months)")
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal"])

with col4:
    property_area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])
    employer_category = st.selectbox("Employer Category", ["Private", "Government"])

st.markdown("---")

# ---------- CONVERT INPUTS ----------
income = to_float(income)
co_income = to_float(co_income)
loan_amount = to_float(loan_amount)
loan_term = to_float(loan_term)
credit_score = to_float(credit_score)
savings = to_float(savings)

# ---------- LOAD MODEL ----------
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    columns = pickle.load(open('columns.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")

# ---------- PREDICTION ----------
if st.button("🔍 Predict Loan Status", use_container_width=True):

    # Validation
    if (
        income == 0 or
        loan_amount == 0 or
        loan_term == 0 or
        credit_score == 0
    ):
        st.warning("⚠️ Please fill all required fields before prediction.")

    else:
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

            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out(cat_cols)
            )

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

# ---------- FIXED COMPACT FOOTER ----------
st.markdown("---")

st.markdown("""
<div style="
    background: linear-gradient(90deg, #1f2937, #111827);
    color: white;
    padding: 15px 20px;
    border-radius: 10px;
    margin-top: 20px;
    font-family: 'Segoe UI', sans-serif;
">

<div style="text-align:center; margin-bottom:8px;">
    <h2 style="margin:0; font-size:24px;">💼 CreditWise</h2>
    <p style="margin:4px 0 0; font-size:15px; color:#d1d5db;">
        AI-Powered Loan Approval & Advisory System
    </p>
</div>

<div style="
    display:flex;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:15px;
    font-size:15px;
    margin-bottom:6px;
">

<div style="margin-left: 18%; color: white">
    <b>Developer Credits :</b><br>
    <span style="color:#d1d5db;">
        Developed by Ashish Gaikar<br>
        Built using Machine Learning with <br> dataset
        analysis in Jupyter Lab
    </span>
</div>

<div style="margin-right: 18%; color:white">
    <b>Dataset Source :</b><br>
    <span style="color:#d1d5db;">
        Standardized
        Loan <br> Data (Synthesized)
    </span>
</div>

</div>

<div style="text-align:center; margin:4px 0;">
    <a href="https://github.com/ashis4" target="_blank"
       style="color:#60a5fa; text-decoration:none; font-size:13px;">
        <img src="https://cdn-icons-png.freepik.com/512/5968/5968866.png" width="30" height="30" style="vertical-align:middle;">
    </a> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
    <a href="https://github.com/ashis4" target="_blank"
       style="color:#60a5fa; text-decoration:none; font-size:13px;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQLYglJ5cooESw4EEKVpBAtJQtGC4qYb9Os_Q&s" width="30" height="30" style="vertical-align:middle;">
    </a>
    &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
    <a href="https://github.com/ashis4" target="_blank"
       style="color:#60a5fa; text-decoration:none; font-size:13px;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSI7wLQhWNc0GUw6OvUbzIUouG-UdxmSqmdMQ&s" width="30" height="30" style="vertical-align:middle;">
    </a>
</div>

<hr style="border:0.5px solid #374151; margin-top: 4px; margin-bottom: 40px;">

<div style="text-align:center; font-size:13px; color:#6b7280; margin:0;">
    © 2026 CreditWise. All rights reserved.
</div>

</div>
""", unsafe_allow_html=True)
