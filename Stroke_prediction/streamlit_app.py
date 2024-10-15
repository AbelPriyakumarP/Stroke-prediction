import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('model.pkl')


st.title('Stroke Prtediction App')

st.divider()

st.write('This app uses Machine Learning for Predicting house price with given features of the house. For using this app you can enter the inputs from this UI and then use predict button.')

st.divider()

gender = st.selectbox('Number of gender', options=[0, 1], index=0)
age = st.number_input('Number of age', min_value=0,max_value=120, value=30)
hypertension = st.number_input('Number of peoples having hypertension', min_value=0, max_value=1, value=0)
heart_disease = st.number_input('Number of peoples having heart_disease', min_value=0,max_value=1 ,value=3)
ever_married = st.selectbox('Number of people ever_married', options=[0, 1], index=0)
work_type = st.selectbox('Number of  peoples worktype',options=['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'])
Residence_type= st.selectbox('Number of  peoples Residence Type', options=['Urban', 'Rural'])
avg_glucose_level=st.number_input('Number of  peoples avg glucose level',min_value=0.0, max_value=300.0, value=100.0)
bmi_smoking_status=st.number_input('Number of  peoples bmi smoking status',min_value=0.0, max_value=60.0, value=25.0)

st.divider() 

X = [[gender,age,hypertension,heart_disease	,ever_married,work_type	,Residence_type,avg_glucose_level,bmi_smoking_status]]
predictbutton = st.button('Predict')

if predictbutton:
    st.snow()
    X_array = np.array(X)
    prediction = model.predict(X_array)[0]
    st.write(f'Price Prediction is {prediction:,.2f}')
else:
    st.write('Please use predict button after entering values')