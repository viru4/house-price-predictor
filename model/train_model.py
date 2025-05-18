import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv('data/housing.csv')

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['Neighborhood', 'Street'], drop_first=True)

# Define features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'GarageCars', 'YearBuilt', 'LotArea'] + \
           [col for col in df.columns if col.startswith('Neighborhood_') or col.startswith('Street_')]

X = df[features]
y = df['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and feature names
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': features}, f)

print("Model trained and saved as model.pkl")
