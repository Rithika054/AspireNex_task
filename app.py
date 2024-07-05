import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the model
model = load_model('customer_churn.h5')

# Streamlit app title
st.title("Customer Churn Prediction")

# Input fields for user data
st.header("Enter Customer Information")

def get_user_input():
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 1)
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.1)

    user_data = {
        'gender': 1 if gender == "Female" else 0,
        'SeniorCitizen': 1 if SeniorCitizen == "Yes" else 0,
        'Partner': 1 if Partner == "Yes" else 0,
        'Dependents': 1 if Dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if PhoneService == "Yes" else 0,
        'MultipleLines': 1 if MultipleLines == "Yes" else 0,
        'InternetService_DSL': 1 if InternetService == "DSL" else 0,
        'InternetService_Fiber optic': 1 if InternetService == "Fiber optic" else 0,
        'InternetService_No': 1 if InternetService == "No" else 0,
        'OnlineSecurity': 1 if OnlineSecurity == "Yes" else 0,
        'OnlineBackup': 1 if OnlineBackup == "Yes" else 0,
        'DeviceProtection': 1 if DeviceProtection == "Yes" else 0,
        'TechSupport': 1 if TechSupport == "Yes" else 0,
        'StreamingTV': 1 if StreamingTV == "Yes" else 0,
        'StreamingMovies': 1 if StreamingMovies == "Yes" else 0,
        'Contract_Month-to-month': 1 if Contract == "Month-to-month" else 0,
        'Contract_One year': 1 if Contract == "One year" else 0,
        'Contract_Two year': 1 if Contract == "Two year" else 0,
        'PaperlessBilling': 1 if PaperlessBilling == "Yes" else 0,
        'PaymentMethod_Electronic check': 1 if PaymentMethod == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if PaymentMethod == "Mailed check" else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if PaymentMethod == "Bank transfer (automatic)" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == "Credit card (automatic)" else 0,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    
    all_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
                   'MultipleLines', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                   'StreamingMovies', 'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
                   'PaperlessBilling', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 
                   'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 
                   'MonthlyCharges', 'TotalCharges']
    
    for column in all_columns:
        if column not in user_data:
            user_data[column] = 0
    
    return pd.DataFrame(user_data, index=[0])

user_input_df = get_user_input()

st.subheader("User Input:")
st.write(user_input_df)

# Scaling
scaler = MinMaxScaler()
user_input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(user_input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Predict churn
if st.button("Predict Churn"):
    try:
        prediction = model.predict(user_input_df)
        churn_prob = prediction[0][0]

        if churn_prob > 0.5:
            st.error(f"High chance of churn: {churn_prob:.2f}")
        else:
            st.success(f"Low chance of churn: {churn_prob:.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
