# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 02:56:54 2024

@author: Adam
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 02:55:48 2024

@author: Adam
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 02:54:46 2024

@author: Adam
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv(r'C:\Users\Adam\Downloads\Final_2022_2024.csv')

def process_first_week_data(df):
    # Convert 'pick_up_date' to datetime
    df['pick_up_date'] = pd.to_datetime(df['pick_up_date'], errors='coerce')

    # Check for any rows that could not be converted
    if df['pick_up_date'].isnull().any():
      

    # Filter for the years 2022 to 2024
        df = df[(df['pick_up_date'].dt.year >= 2022) & (df['pick_up_date'].dt.year <= 2024)]

    # Add month and day columns
    df['month'] = df['pick_up_date'].dt.month
    df['day'] = df['pick_up_date'].dt.day

    # Keep only records from the first week of each month (1st to 7th)
    df = df[(df['pick_up_date'].dt.day >= 15) & (df['pick_up_date'].dt.day <= 21)]


    # Group by month and truck type, then sum the number of pallets
    delivery_volume = df.groupby(['month', 'trucking'])['number_of_pallets_per_day'].sum().reset_index()

    return delivery_volume

# Process the data
third_week_volume = process_first_week_data(data)

# Check the processed data before plotting
print("Delivery Volume Data:")
print(third_week_volume)

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(data=third_week_volume, x='month', y='number_of_pallets_per_day', hue='trucking')
plt.title('Volume of Deliveries by Truck Type (Third Week of Each Month from 2022 to 2024)')
plt.xlabel('Month')
plt.ylabel('Number of Pallets')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Truck Type')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
