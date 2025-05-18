# house_price_predictor/app.py

import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# App title
st.title("\U0001F3E0 House Price Predictor")

st.markdown("""
This app predicts the **price of a house** based on user input features like living area, number of bedrooms, bathrooms, and garage cars.
""")

# Input fields
area = st.number_input('Living Area (in square feet)', min_value=100, max_value=10000, step=50)
bedrooms = st.slider('Number of Bedrooms', 1, 10)
bathrooms = st.slider('Number of Bathrooms', 1, 10)
garage = st.slider('Number of Garage Cars', 0, 5)

# Prediction
if st.button('Predict House Price'):
    features = np.array([[area, bedrooms, bathrooms, garage]])
    prediction = model.predict(features)
    st.success(f'Estimated House Price: ${prediction[0]:,.2f}')
