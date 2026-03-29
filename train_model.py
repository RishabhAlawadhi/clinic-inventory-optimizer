import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# 1. Load the Data
print("Loading data...")
df = pd.read_csv('clinic_inventory_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 2. Feature Engineering (The Secret Sauce)
# We need to give the model historical context so it understands trends
df['Day_of_Year'] = df['Date'].dt.dayofyear
df['Month'] = df['Date'].dt.month

# Create lag features: What was the exact demand 7 and 14 days ago?
df['Demand_Lag_7'] = df.groupby('Item_Name')['Quantity_Dispensed'].shift(7)
df['Demand_Lag_14'] = df.groupby('Item_Name')['Quantity_Dispensed'].shift(14)

# Drop the initial rows that now have NaN values because we shifted the data
df = df.dropna()

# 3. Prepare for Training
# Convert the text 'Item_Name' into numbers the model can understand (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['Item_Name'])

# Define our input features (X) and the target we want to predict (y)
features = [col for col in df_encoded.columns if col not in ['Date', 'Quantity_Dispensed']]
X = df_encoded[features]
y = df_encoded['Quantity_Dispensed']

# Split into training (80%) and testing (20%) sets
# We set shuffle=False to keep the data in chronological order for time-series validity
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train the XGBoost Model
print("Training XGBoost Regressor...")
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the Model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"\nModel Mean Absolute Error: {mae:.2f} units")
print(f"This means our predictions are off by an average of just {round(mae, 1)} items per day.")

# 6. Save the Model and Features
joblib.dump(model, 'inventory_model.pkl')
joblib.dump(features, 'model_features.pkl')
print("\nSuccess! Model saved as 'inventory_model.pkl'.")
