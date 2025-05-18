import streamlit as st
import pickle
import pandas as pd

# Load model
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    feature_names = data['features']

st.title("🏠 House Price Predictor")

# Get neighborhoods and street types from feature names
neighborhoods = sorted([col.replace("Neighborhood_", "") for col in feature_names if col.startswith("Neighborhood_")])
streets = sorted([col.replace("Street_", "") for col in feature_names if col.startswith("Street_")])

# Input fields
area = st.number_input('Living Area (GrLivArea in sq ft)', min_value=100, max_value=10000, step=50)
lot_area = st.number_input('Lot Area (sq ft)', min_value=500, max_value=100000, step=100)
bedrooms = st.slider('Bedrooms (BedroomAbvGr)', 1, 10)
bathrooms = st.slider('Bathrooms (FullBath)', 1, 5)
garage = st.slider('Garage Capacity (GarageCars)', 0, 5)
year_built = st.number_input('Year Built', min_value=1800, max_value=2025, step=1)
location = st.selectbox("Neighborhood", neighborhoods)
street = st.selectbox("Street Type", streets)

# Prepare input
input_data = {
    'GrLivArea': area,
    'LotArea': lot_area,
    'BedroomAbvGr': bedrooms,
    'FullBath': bathrooms,
    'GarageCars': garage,
    'YearBuilt': year_built,
}

# Encode Neighborhood and Street
for n in neighborhoods:
    input_data[f'Neighborhood_{n}'] = 1 if n == location else 0

for s in streets:
    input_data[f'Street_{s}'] = 1 if s == street else 0

# Create DataFrame
input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Predict
if st.button("Predict House Price"):
    prediction = model.predict(input_df)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")
