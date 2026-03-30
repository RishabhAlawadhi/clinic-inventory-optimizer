# 🏥 AI-Powered Clinic Inventory Optimizer

## 📌 The Problem
Local clinics and pharmacies often struggle with inventory management. Relying on static spreadsheets or manual guessing leads to critical shortages of essential medications during seasonal spikes (like Flu season) or expensive waste when overstocked items expire.

## 💡 The Solution
This project is an end-to-end Machine Learning decision support system that predicts the 7-day future demand for medical supplies. It uses historical dispensing data, local weather patterns, and seasonality to forecast exact inventory needs, wrapped in an interactive UI for clinic managers.

## 🛠️ Tech Stack
* **Data Engineering:** Python, Pandas, NumPy (Custom synthetic data generation mimicking real-world clinic stochastic operations)
* **Machine Learning:** XGBoost Regressor (Time-series forecasting, lag feature engineering)
* **Frontend/UI:** Streamlit (Interactive dashboard, real-time prediction updates)

## 🧠 Business Logic Engine
The dashboard doesn't just display predictions; it actively evaluates current stock against AI forecasts to provide actionable alerts:
* **🚨 Critical Shortage:** Triggers if current stock is dangerously low compared to predicted daily demand.
* **⚠️ High Waste Risk:** Flags items where current stock vastly exceeds the predicted demand.
* **✅ Optimal Stock:** Confirms when supply safely meets expected demand.

## 🚀 How to Run Locally

1. Clone the repository
2. Install dependencies: `pip install pandas numpy xgboost scikit-learn streamlit`
3. Generate the synthetic dataset: `python generate_data.py`
4. Train the XGBoost model: `python train_model.py`
5. Launch the interactive dashboard: `python -m streamlit run app.py`
