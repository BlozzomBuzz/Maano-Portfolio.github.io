# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 06:08:36 2024

@author: Adam
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load data
nb = pd.read_csv(r"C:\Users\Adam\Downloads\CAPSTONE\Final2021_2024_withID.csv")

# Handle missing values if necessary
nb.dropna(inplace=True)

# Convert date columns to datetime and extract relevant features
nb['pick_up_date'] = pd.to_datetime(nb['pick_up_date'])
nb['year'] = nb['pick_up_date'].dt.year
nb['month'] = nb['pick_up_date'].dt.month
nb['day'] = nb['pick_up_date'].dt.day
nb['day_of_week'] = nb['pick_up_date'].dt.dayofweek

# Drop unnecessary columns
nb.drop(columns=['pick_up_date', 'shipment_id'], inplace=True)

# Create billing categories
def categorize_billing(billing):
    if billing < 85:
        return 'Low'
    elif billing < 120:
        return 'Medium'
    else:
        return 'High'

nb['billing_category'] = nb['trucking_billing'].apply(categorize_billing)

# Encode billing categories
label_encoder = LabelEncoder()
nb['billing_category_encoded'] = label_encoder.fit_transform(nb['billing_category'])

# Identify categorical columns for encoding
categorical_columns = ['forwarder', 'warehouse_location', 'consignee', 'destination', 'trucking']

# Encode categorical features
for column in categorical_columns:
    le = LabelEncoder()
    nb[column] = le.fit_transform(nb[column].astype(str))  # Convert to string to avoid errors

# Select relevant features for the correlation matrix
features = nb[['day', 'month', 'year', 'day_of_week', 
                'forwarder', 'warehouse_location', 'consignee', 
                'destination', 'number_of_packages', 'weight', 
                'quantity', 'number_of_pallets_per_day', 'trucking', 
                'billing_category_encoded']]

# Calculate the correlation matrix
correlation_matrix = features.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(16, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True, 
            mask=np.triu(correlation_matrix))  # Mask the upper triangle for better readability

plt.title('Correlation Heatmap of Features and Billing Categories', fontsize=16)
plt.show()

# Create a box plot to visualize the distribution of billing categories across number of pallets per day
plt.figure(figsize=(14, 7))
sns.boxplot(data=nb, x='number_of_pallets_per_day', y='billing_category', palette='Set2')

# Add titles and labels
plt.title('Distribution of Billing Categories by Number of Pallets per Day', fontsize=16, fontweight='bold')
plt.xlabel('Number of Pallets per Day', fontsize=12)
plt.ylabel('Billing Category', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability
plt.tight_layout()

# Show plot
plt.show()

# Create a violin plot to visualize the distribution of billing categories across number of pallets per day
plt.figure(figsize=(14, 7))
sns.violinplot(data=nb, x='number_of_pallets_per_day', y='billing_category', palette='Set2', inner='quartile')

# Add titles and labels
plt.title('Distribution of Billing Categories by Number of Pallets per Day', fontsize=16, fontweight='bold')
plt.xlabel('Number of Pallets per Day', fontsize=12)
plt.ylabel('Billing Category', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability
plt.tight_layout()

# Show plot
plt.show()
