pip install -r requirements.txt

import pickle
import pandas as pd
import streamlit as st


with(open("customer_churn_model.pkl", "rb")) as f:
  model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["feature_names"]

'''
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

'''


with open("label_encoder.pkl", "rb") as f:
  encoders = pickle.load(f)


# page title
st.title('Customer churn Prediction using ML')

# getting the input data from the user
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.text_input('gender')

with col2:
    seniorCitizen = st.text_input('SeniorCitizen')

with col3:
    partner = st.text_input('Partner')

with col1:
    dependents = st.text_input('Dependents')

with col2:
    tenure = st.text_input('tenure')

with col3:
    phoneService = st.text_input('PhoneService')

with col1:
    multipleLines = st.text_input('MultipleLines')

with col2:
    internetService = st.text_input('InternetService')

with col3:
    OnlineBackup = st.text_input('OnlineBackup')

with col1:
    DeviceProtection = st.text_input('DeviceProtection')

with col2:
    TechSupport = st.text_input('TechSupport')

with col3:
    StreamingTV = st.text_input('StreamingTV')

with col1:
    StreamingMovies = st.text_input('StreamingMovies')

with col2:
    Contract = st.text_input('Contract')

with col3:
    PaperlessBilling = st.text_input('PaperlessBilling')

with col1:
    PaymentMethod = st.text_input('PaymentMethod')

with col2:
    MonthlyCharges = st.text_input('MonthlyCharges')

with col3:
    TotalCharges = st.text_input('TotalCharges')




# code for Prediction
customer_churn = ''

# creating a button for Prediction

if st.button('Diabetes Test Result'):

    user_input = [
    gender, seniorCitizen, partner, dependents, tenure, phoneService, 
    multipleLines, internetService, OnlineBackup, DeviceProtection, 
    TechSupport, StreamingTV, StreamingMovies, Contract, 
    PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
    ] 

    input_data_df = pd.DataFrame([user_input])

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
