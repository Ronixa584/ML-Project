import streamlit as st
import joblib
import pandas as pd

st.write("# The Crop Prdiction using the ML")

# getting user input
N = st.number_input("Enter value N(ratio of Nitrogen content in soil)")
P = st.number_input("Enter value P(ratio of Phosphorous content in soil)")
K = st.number_input("Enter value K(ratio of Potassium content in soil)")
temperature = st.number_input("Enter temperature(in degree Celsius)")
humidity = st.number_input("Enter humidity(relative in %)")
ph = st.number_input("Enter PH value of the soil")
rainfall = st.number_input("Enter Rainfall(in mm)")


df_pred = pd.DataFrame([[N, P,K,temperature, humidity, ph, rainfall]],columns= ['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall'])

#Calling the ML model to make prediction
model = joblib.load('fhs_rf_model.pkl')
prediction = model.predict(df_pred)

# This below code is done to align button center
col1, col2, col3 , col4, col5 = st.columns(5)
with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    if st.button('Predict') and N!=0 and P!=0 and K!=0 and temperature!=0 and humidity!=0 and ph!=0 and rainfall!=0 :
        st.write(prediction)