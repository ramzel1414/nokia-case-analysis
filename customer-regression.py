import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data for years 2018-2023 and their respective values
data = {
    'Year': [2018, 2019, 2020, 2021, 2022, 2023],
    'Communication Service Providers': [1167, 1409, 1502, 1575, 2007, 2282],
    'Enterprise': [449, 447, 440, 440, 440, 440],
    'Licenses': [1148, 1391, 1571, 1575, 2007, 2282],
    'Other': [925, 1148, 1402, 1571, 1502, 1575],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Prepare the feature 'Year' as the independent variable (X)
X = df[['Year']]  # Only the 'Year' column as the independent variable

# Prepare the targets (dependent variables) for prediction
y_communication = df['Communication Service Providers']
y_enterprise = df['Enterprise']
y_licenses = df['Licenses']
y_other = df['Other']

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to each target and make predictions for 2024-2028
future_years = np.array([2024, 2025, 2026, 2027, 2028]).reshape(-1, 1)

# Predictions for Communication Service Providers
model.fit(X, y_communication)
pred_communication = model.predict(future_years)

# Predictions for Enterprise
model.fit(X, y_enterprise)
pred_enterprise = model.predict(future_years)

# Predictions for Licenses
model.fit(X, y_licenses)
pred_licenses = model.predict(future_years)

# Predictions for Other
model.fit(X, y_other)
pred_other = model.predict(future_years)

# Output the predictions for the next 5 years (2024-2028)
print("Predictions for Communication Service Providers (2024-2028):")
for year, value in zip(future_years.flatten(), pred_communication):
    print(f"Year {year}: {value:.2f}")

print("\nPredictions for Enterprise (2024-2028):")
for year, value in zip(future_years.flatten(), pred_enterprise):
    print(f"Year {year}: {value:.2f}")

print("\nPredictions for Licenses (2024-2028):")
for year, value in zip(future_years.flatten(), pred_licenses):
    print(f"Year {year}: {value:.2f}")

print("\nPredictions for Other (2024-2028):")
for year, value in zip(future_years.flatten(), pred_other):
    print(f"Year {year}: {value:.2f}")

# Plot the actual vs predicted values for Communication Service Providers
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.scatter(df['Year'], y_communication, color='blue', label='Actual')
plt.plot(future_years, pred_communication, color='red', label='Predicted')
plt.title('Communication Service Providers: Actual vs Predicted')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()

# Plot the actual vs predicted values for Enterprise
plt.subplot(2, 2, 2)
plt.scatter(df['Year'], y_enterprise, color='blue', label='Actual')
plt.plot(future_years, pred_enterprise, color='red', label='Predicted')
plt.title('Enterprise: Actual vs Predicted')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()

# Plot the actual vs predicted values for Licenses
plt.subplot(2, 2, 3)
plt.scatter(df['Year'], y_licenses, color='blue', label='Actual')
plt.plot(future_years, pred_licenses, color='red', label='Predicted')
plt.title('Licenses: Actual vs Predicted')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()

# Plot the actual vs predicted values for Other
plt.subplot(2, 2, 4)
plt.scatter(df['Year'], y_other, color='blue', label='Actual')
plt.plot(future_years, pred_other, color='red', label='Predicted')
plt.title('Other: Actual vs Predicted')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()