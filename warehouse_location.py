import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

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

# Encode billing categories and warehouse locations
label_encoder = LabelEncoder()
nb['billing_category_encoded'] = label_encoder.fit_transform(nb['billing_category'])
nb['warehouse_location_encoded'] = label_encoder.fit_transform(nb['warehouse_location'])

# Prepare features and target variable
X = nb[['warehouse_location_encoded', 'weight', 'number_of_packages', 'quantity', 'number_of_pallets_per_day']]
y = nb['billing_category_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Create a pipeline with a Random Forest Classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameters for tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
}

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_balanced, y_train_balanced)

# Best model from grid search
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print('Classification Report:')
print(report)

# Visualization of feature importances
importances = best_model.named_steps['classifier'].feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.show()
