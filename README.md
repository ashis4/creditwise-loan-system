# 🏦 CreditWise - AI Loan Approval & Advisory System
CreditWise is a machine learning-based web application that predicts loan approval status and provides intelligent financial suggestions. The system analyzes applicant details and assists in making data-driven loan decisions.

## 🚀 Live Demo
🔗 https://creditwise-loan-system-aqxr7swfz89eqoljvynggs.streamlit.app/

## 📌 Project Overview
This project demonstrates the complete pipeline of a real-world machine learning system:
- Data preprocessing
- Feature engineering
- Model training
- Model deployment using Streamlit

The application allows users to input financial and personal details and get instant loan approval predictions along with suggestions.

## 🔄 Project Workflow

### 1️⃣ Data Collection
- Used structured loan dataset containing applicant financial details
- Features include income, credit score, employment status, etc.

### 2️⃣ Data Preprocessing
- Handled missing values
- Converted categorical variables into numerical format
- Normalized numerical data using scaling
- Ensured dataset consistency

### 3️⃣ Feature Engineering
- Created new feature: **DTI Ratio (Debt-to-Income Ratio)**
- Generated polynomial features:
  - Credit Score²
  - DTI²
- Improved model performance with derived features

### 4️⃣ Model Building
- Trained machine learning model (classification)
- Used Scikit-learn
- Evaluated model performance

### 5️⃣ Model Optimization
- Feature importance analysis
- Improved model reliability
- Tuned model for better predictions

### 6️⃣ Model Deployment
- Built interactive UI using Streamlit
- Saved model using Pickle
- Integrated:
  - model.pkl
  - scaler.pkl
  - encoder.pkl
  - columns.pkl

### 7️⃣ Prediction & Advisory System
- Predicts loan approval status
- Displays confidence score
- Provides:
  - Reasons for rejection
  - Suggestions for improvement

## 💡 Key Features

- 🔍 Real-time loan prediction  
- 💰 EMI calculation  
- 📊 Confidence score display  
- ❗ Rejection reasons  
- 💡 Smart financial suggestions  
- 🎯 Clean and user-friendly UI  

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Jupyter Notebook  

## 📁 Project Structure
