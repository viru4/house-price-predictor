# 🏠 House Price Predictor

A machine learning app to predict house prices using features like area, location, year built, and more.

## 🚀 Features
- Input via web app (Streamlit)
- Uses Random Forest Regressor
- Trained on Kaggle House Prices dataset
- Supports features:
  - Living area
  - Lot area
  - Bedrooms
  - Bathrooms
  - Garage size
  - Year built
  - Neighborhood
  - Street type

## 🛠 How to Run

```bash
pip install -r requirements.txt
python model/train_model.py
streamlit run app.py
## link of website
https://house-price-predictor-4.streamlit.app/
