import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# Create a DataFrame to visualize the average trucking billing by year
average_billing_by_year = nb.groupby('year')['trucking_billing'].mean().reset_index()

# Fit linear regression model
X = average_billing_by_year['year'].values.reshape(-1, 1)
y = average_billing_by_year['trucking_billing'].values

model = LinearRegression()
model.fit(X, y)

# Predict for future years (e.g., 2025 to 2030)
future_years = np.array(range(2025, 2031)).reshape(-1, 1)
predictions = model.predict(future_years)

# Calculate the metrics for the historical average trucking billing predictions
historical_predictions = model.predict(X)

# Evaluate metrics
mae = mean_absolute_error(y, historical_predictions)
mse = mean_squared_error(y, historical_predictions)
r2 = r2_score(y, historical_predictions)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R²): {r2:.2f}')

# Plotting
plt.figure(figsize=(12, 6))

# Actual average trucking billing
sns.lineplot(data=average_billing_by_year, x='year', y='trucking_billing', marker='o', label='Average Trucking Billing')

# Predicted trucking billing for future years
plt.plot(future_years, predictions, color='red', linestyle='--', marker='o', label='Predicted Trucking Billing')

plt.title('Average Trucking Billing by Year with Predictions')
plt.xlabel('Year')
plt.ylabel('Average Trucking Billing')
plt.xticks(list(average_billing_by_year['year']) + list(future_years.flatten()), rotation=45)  # Rotate x labels for better readability
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
