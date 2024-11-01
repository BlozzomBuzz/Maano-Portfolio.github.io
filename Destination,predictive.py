import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
nb = pd.read_csv(r"C:\Users\Adam\Downloads\CAPSTONE\Final2021_2024_withID.csv")

# Handle missing values
nb.dropna(inplace=True)

# Convert date columns to datetime and extract relevant features
nb['pick_up_date'] = pd.to_datetime(nb['pick_up_date'])
nb['year'] = nb['pick_up_date'].dt.year
nb['month'] = nb['pick_up_date'].dt.month
nb['day'] = nb['pick_up_date'].dt.day
nb['day_of_week'] = nb['pick_up_date'].dt.dayofweek

# Drop unnecessary columns
nb.drop(columns=['pick_up_date', 'shipment_id'], inplace=True)

# Prepare features and target variable
X = nb[['year', 'month', 'day', 'day_of_week']]
y = nb['trucking_billing']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Create a DataFrame to visualize the predictions along with actual billing
predictions_df = pd.DataFrame({'Actual Billing': y_test, 'Predicted Billing': y_pred})

# Calculate average predicted billing by destination
predictions_df['Destination'] = nb.loc[y_test.index, 'destination']
average_predicted_billing = predictions_df.groupby('Destination').mean().reset_index()

# Sort values for better visualization
average_predicted_billing.sort_values(by='Predicted Billing', ascending=False, inplace=True)

# Set plot size and style
plt.figure(figsize=(14, 8))
sns.barplot(data=average_predicted_billing, x='Destination', y='Predicted Billing', palette='viridis')
plt.title('Predicted Average Trucking Billing by Destination')
plt.xlabel('Destination')
plt.ylabel('Predicted Average Trucking Billing')
plt.xticks(rotation=45)  # Rotate x labels for better readability
plt.grid(axis='y')
plt.tight_layout()

# Show the plot
plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
r_squared = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R²): {r_squared:.2f}')
