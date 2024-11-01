import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load data
nb = pd.read_csv(r"C:\Users\Adam\Downloads\CAPSTONE\Final2021_2024_withID.csv")

# Handle missing values if necessary
nb.dropna(inplace=True)

# Convert date columns to datetime and extract relevant features
nb['pick_up_date'] = pd.to_datetime(nb['pick_up_date'])
nb['year'] = nb['pick_up_date'].dt.year
nb['month'] = nb['pick_up_date'].dt.month  # Extract month
nb['day'] = nb['pick_up_date'].dt.day
nb['day_of_week'] = nb['pick_up_date'].dt.dayofweek  # Monday=0, Sunday=6

# Drop unnecessary columns
nb.drop(columns=['pick_up_date', 'shipment_id'], inplace=True)

# Define target variable as trucking_billing (continuous variable)
target = nb['trucking_billing']  # Continuous variable for regression

# Identify categorical columns for encoding
categorical_columns = ['forwarder', 'warehouse_location', 'consignee', 'destination', 'day_of_week', 'trucking']

# Encode categorical features
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    nb[column] = le.fit_transform(nb[column].astype(str))  # Convert to string to avoid errors
    label_encoders[column] = le

# Select features for modeling (excluding trucking_billing)
features = nb[['day', 'month', 'year', 'day_of_week', 
                'forwarder', 'warehouse_location', 'consignee', 
                'destination', 'number_of_packages', 'weight', 
                'quantity', 'number_of_pallets_per_day', 'trucking']]

# Calculate the correlation matrix
correlation_matrix = features.copy()
correlation_matrix['trucking_billing'] = target  # Add target variable to features DataFrame

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(16, 10))
sns.heatmap(correlation_matrix.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True, 
            mask=np.triu(correlation_matrix.corr()))  # Mask the upper triangle for better readability

plt.title('Correlation Heatmap of Features and Trucking Billing')
plt.show()





