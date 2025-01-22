
import pickle
import pandas as pd
import streamlit as st


with(open("customer_churn_model.pkl", "rb")) as f:
  model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["feature_names"]



with open("label_encoder.pkl", "rb") as f:
  encoders = pickle.load(f)


# page title
st.title('Customer churn Prediction using ML')

# getting the input data from the user
col1, col2, col3 = st.columns(3)


# Create a dropdown

with col1:
    options = ["Male", "Female"]    
    gender = st.selectbox(
    "gender", options
    )

with col2:
    options = [1, 0]    
    seniorCitizen = st.selectbox(
    "SeniorCitizen", options
    )

with col3:
    options = ["Yes", "No"]    
    partner = st.selectbox(
    "Partner", options
    )

with col1:
    options = ["Yes", "No"]
    dependents = st.selectbox(
    "Dependents", options
    )

with col2:
    tenure = st.text_input('tenure')

with col3:
    options = ["Yes", "No"]
    phoneService = st.selectbox(
    "PhoneService", options
    )

with col1:
    options = ["No phone service", "No"]
    multipleLines = st.selectbox(
    "MultipleLines", options
    )

with col2:
    options = ['DSL', 'Fiber optic', 'No']
    internetService = st.selectbox(
    "InternetService", options
    )

with col3:
    options = ["Yes", "No", "No internet service"]
    OnlineBackup = st.selectbox(
    "OnlineBackup", options
    )

with col1:
    options = ["Yes", "No", "No internet service"]
    DeviceProtection = st.selectbox(
    "DeviceProtection", options
    )

with col2:
    options = ["Yes", "No"]
    TechSupport = st.selectbox(
    "TechSupport", options
    )

with col3:
    options = ["Yes", "No"]
    StreamingTV = st.selectbox(
    "StreamingTV", options
    )

with col1:
    options = ["Yes", "No"]
    StreamingMovies = st.selectbox(
    "StreamingMovies", options
    )

with col2:
    options = ["Month-to-month", "One year"]
    Contract = st.selectbox(
    "Contract", options
    )

with col3:
    options = ["Yes", "No"]
    PaperlessBilling = st.selectbox(
    "PaperlessBilling", options
    )

with col1:
    options = ["Electronic check", "Mailed check", "Bank transfer (automatic)"]
    PaymentMethod = st.selectbox(
    "PaymentMethod", options
    )

with col2:
    options = ["Yes", "No"]
    OnlineSecurity = st.selectbox(
    "OnlineSecurity", options
    )

with col3:
    MonthlyCharges = st.text_input('MonthlyCharges')

with col1:
    TotalCharges = st.text_input('TotalCharges')


# code for Prediction
customer_churn = ''
pred_prob = ''
# creating a button for Prediction

if st.button('Customer Churn Result'):

    user_input = [
    gender, seniorCitizen, partner, dependents, tenure, phoneService, 
    multipleLines, internetService, OnlineSecurity, OnlineBackup, DeviceProtection, 
    TechSupport, StreamingTV, StreamingMovies, Contract, 
    PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
    ]

    columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ] 

    input_data_df = pd.DataFrame([user_input],columns=columns)

    # encode categorical featires using teh saved encoders
    for column, encoder in encoders.items():
        input_data_df[column] = encoder.transform(input_data_df[column])


    # make a prediction
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)

    if prediction[0] == 1:
        customer_churn = 'The customer will churn'
    else:
        customer_churn = 'The customer will not churn'

st.success(customer_churn)
st.success(pred_prob)
