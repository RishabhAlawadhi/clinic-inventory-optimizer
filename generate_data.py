import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Setup Parameters
np.random.seed(42) # For reproducibility
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date)

# Define our clinic items and their baseline daily demand
items = {
    'Amoxicillin': {'base': 15, 'noise': 5},
    'Paracetamol': {'base': 30, 'noise': 10},
    'Flu Vaccines': {'base': 2, 'noise': 2},
    'Cough Syrup': {'base': 10, 'noise': 4},
    'Bandages': {'base': 8, 'noise': 3}
}

data = []

# 2. Generate Daily Data
for current_date in date_range:
    month = current_date.month
    day_of_week = current_date.weekday() # 0 = Monday, 6 = Sunday
    
    # Simulate seasonality (Flu season is roughly Nov - Feb)
    is_flu_season = 1 if month in [11, 12, 1, 2] else 0
    
    # Simulate temperature (colder in winter months)
    if is_flu_season:
        temp = np.random.normal(10, 5) # Mean 10C, Std 5
    elif month in [6, 7, 8]:
        temp = np.random.normal(30, 4) # Mean 30C, Std 4
    else:
        temp = np.random.normal(20, 6) # Spring/Fall
        
    for item_name, stats in items.items():
        # Start with baseline demand + random noise
        demand = np.random.normal(stats['base'], stats['noise'])
        
        # Inject Business Logic / "Messiness"
        # 1. Flu season spikes demand for specific items
        if is_flu_season == 1:
            if item_name == 'Flu Vaccines':
                demand += np.random.normal(25, 5) 
            elif item_name == 'Cough Syrup':
                demand += np.random.normal(15, 5)
                
        # 2. Weekends are slower (Clinics might be closed or have limited hours on Sunday)
        if day_of_week == 6: # Sunday
            demand = demand * 0.2
        elif day_of_week == 5: # Saturday
            demand = demand * 0.6
            
        # Ensure we don't have negative dispensing (can't sell -3 bandages)
        demand = max(0, int(demand))
        
        # Append to our dataset
        data.append({
            'Date': current_date,
            'Item_Name': item_name,
            'Day_of_Week': day_of_week,
            'Local_Temperature': round(temp, 1),
            'Is_Flu_Season': is_flu_season,
            'Quantity_Dispensed': demand
        })

# 3. Create DataFrame and Save
df = pd.DataFrame(data)

# Sort by Date and Item for a clean time-series format
df = df.sort_values(by=['Date', 'Item_Name']).reset_index(drop=True)

# Save to CSV so you can load it into your ML pipeline later
df.to_csv('clinic_inventory_data.csv', index=False)

print("Dataset generated successfully!")
print(f"Total records: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())