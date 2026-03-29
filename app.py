import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="Clinic AI Inventory", layout="wide")
st.title("🏥 AI-Powered Clinic Inventory Optimizer")
st.markdown("Predictive analytics for medical supplies to prevent waste and shortages.")

# 2. Load the AI Model
@st.cache_resource
def load_model():
    model = joblib.load('inventory_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

try:
    model, expected_features = load_model()
except FileNotFoundError:
    st.error("Model files not found. Please wait for train_model.py to finish running.")
    st.stop()

# 3. Sidebar Inputs (The User Interface)
st.sidebar.header("Forecast Parameters")
forecast_date = st.sidebar.date_input("Target Date", datetime.today())
temperature = st.sidebar.slider("Local Temperature (°C)", -10, 45, 20)
is_flu_season = st.sidebar.checkbox("Is it Flu Season?", value=False)

# 4. Simulated Clinic Database
# In a real app, this would connect to a SQL database. Here we hardcode current stock.
current_stock = {
    'Amoxicillin': 45,
    'Paracetamol': 300,
    'Flu Vaccines': 10,
    'Cough Syrup': 80,
    'Bandages': 150
}

st.write(f"### 🔮 Demand Forecast for {forecast_date.strftime('%B %d, %Y')}")

# 5. Generate Predictions
items = list(current_stock.keys())
predictions = {}

for item in items:
    # Create a single row of data formatted exactly how the model expects it
    input_data = pd.DataFrame(columns=expected_features)
    input_data.loc[0] = 0 
    
    # Fill in the user's inputs
    input_data['Day_of_Year'] = forecast_date.timetuple().tm_yday
    input_data['Month'] = forecast_date.month
    input_data['Day_of_Week'] = forecast_date.weekday()
    input_data['Local_Temperature'] = temperature
    input_data['Is_Flu_Season'] = 1 if is_flu_season else 0
    
    # Simulate historical lag data (normally fetched from SQL)
    input_data['Demand_Lag_7'] = np.random.randint(5, 20) 
    input_data['Demand_Lag_14'] = np.random.randint(5, 20)
    
    # Tell the model which specific item we are asking about
    item_col = f"Item_Name_{item}"
    if item_col in expected_features:
        input_data[item_col] = 1
        
    # Ensure correct column order and predict
    input_data = input_data[expected_features]
    pred = model.predict(input_data)[0]
    predictions[item] = max(0, int(pred)) # Demand can't be negative

# 6. Display the Dashboard & Optimization Logic
cols = st.columns(len(items))

for i, (item, pred_demand) in enumerate(predictions.items()):
    stock = current_stock[item]
    
    with cols[i]:
        st.subheader(item)
        # Compare AI Prediction vs Reality
        st.metric("Predicted Daily Demand", pred_demand)
        st.write(f"**Current Stock:** {stock}")
        
        # The Business Logic Engine
        if stock < pred_demand * 2: # If we have less than 2 days of supply
            st.error("🚨 Critical Shortage")
            if st.button(f"Generate PO", key=item):
                st.success("Purchase Order Sent to Supplier!")
        elif stock > pred_demand * 10: # If we have too much supply
            st.warning("⚠️ High Waste Risk")
        else:
            st.success("✅ Stock Optimal")

st.markdown("---")
st.markdown("*This dashboard uses an XGBoost machine learning model trained on historical clinic data to predict inventory needs based on seasonality, weather, and past trends.*")