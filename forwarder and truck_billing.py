import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load data
nb = pd.read_csv(r"C:\Users\Adam\Downloads\CAPSTONE\Final2021_2024_withID.csv")

# Handle missing values
nb.dropna(inplace=True)

# Convert date columns to datetime
nb['pick_up_date'] = pd.to_datetime(nb['pick_up_date'])

# Create billing categories
def categorize_billing(billing):
    if 14 <= billing <= 85:
        return 'Low'
    elif 86 <= billing <= 120:
        return 'Medium'
    else:
        return 'High'

nb['billing_category'] = nb['trucking_billing'].apply(categorize_billing)

# Encode forwarders to numeric for regression analysis
forwarder_encoded = {name: idx for idx, name in enumerate(nb['forwarder'].unique())}
nb['forwarder_encoded'] = nb['forwarder'].map(forwarder_encoded)

# Prepare additional features for regression
nb['month'] = nb['pick_up_date'].dt.month
nb['day_of_week'] = nb['pick_up_date'].dt.dayofweek

# Select features and target variable
features = ['forwarder_encoded', 'month', 'day_of_week', 'number_of_packages', 'weight', 'quantity', 'number_of_pallets_per_day']
X = nb[features]
y = nb['trucking_billing']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R²): {r2:.2f}')

# Generate predictions for the average billing by forwarder
average_billing_by_forwarder = (
    nb.groupby(['forwarder', 'billing_category'])['trucking_billing']
    .mean()
    .reset_index()
)

# Prepare an array to hold predicted values
predicted_billing = []

# Loop over the unique forwarders to make predictions
for forwarder in average_billing_by_forwarder['forwarder']:
    forwarder_idx = forwarder_encoded[forwarder]
    # Create a placeholder for features (using mean values where necessary)
    mean_values = {
        'month': nb['month'].mean(),
        'day_of_week': nb['day_of_week'].mean(),
        'number_of_packages': nb['number_of_packages'].mean(),
        'weight': nb['weight'].mean(),
        'quantity': nb['quantity'].mean(),
        'number_of_pallets_per_day': nb['number_of_pallets_per_day'].mean()
    }
    # Prepare the input for the model
    input_features = np.array([[forwarder_idx, mean_values['month'], mean_values['day_of_week'], 
                                 mean_values['number_of_packages'], mean_values['weight'], 
                                 mean_values['quantity'], mean_values['number_of_pallets_per_day']]])
    predicted_billing.append(model.predict(input_features)[0])

# Add predicted billing to the average billing DataFrame
average_billing_by_forwarder['predicted_billing'] = predicted_billing

# Set plot size and style
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Create a bar plot for average billing by forwarder and billing category
bar_plot = sns.barplot(data=average_billing_by_forwarder, 
                       x='forwarder', 
                       y='trucking_billing', 
                       hue='billing_category', 
                       palette='Set2', 
                       alpha=0.8,
                       hue_order=['High', 'Medium', 'Low'])  # Specify hue order

# Add predictions to the plot
plt.plot(average_billing_by_forwarder['forwarder'], average_billing_by_forwarder['predicted_billing'], 
         color='darkred', linestyle='--', linewidth=2, label='Predicted Trend')

# Add titles and labels
plt.title('Average Trucking Billing by Forwarder with Improved Predictions', fontsize=16, fontweight='bold')
plt.xlabel('Forwarder', fontsize=12)
plt.ylabel('Average Trucking Billing ($)', fontsize=12)
plt.xticks(rotation=30, ha='right')  # Rotate x labels for better readability
plt.legend(title='Billing Category', title_fontsize='12', fontsize='10', loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better readability

# Show plot
plt.tight_layout()  # Adjust layout for better spacing
plt.show()
