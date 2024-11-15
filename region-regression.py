import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Data for regions over the years (from 2018 to 2023)
data = {
    'Region': ['Asia Pacific', 'Europe', 'Greater China', 'India', 'Latin America', 'Middle East & Africa', 'North America'],
    'Q4\'23': [683, 1533, 337, 379, 322, 646, 1518],
    'Q4\'22': [801, 2351, 356, 568, 387, 595, 2070],
    'Q4\'21': [712, 1940, 406, 250, 350, 610, 2146],
    'Q4\'20': [806, 1789, 459, 294, 330, 570, 2304],
    'Q4\'19': [1383, 1895, 469, None, 467, 619, 2070],  # Note: India has missing data
    'Q4\'18': [1189, 1916, 622, None, 452, 564, 2126],  # Note: India has missing data
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Prepare years (2018-2023)
years = np.array([2018, 2019, 2020, 2021, 2022, 2023])

# Initialize a list to store the combined data
combined_data = []

# Map regions to numerical values (1 for Asia Pacific, 2 for Europe, etc.)
region_map = {
    'Asia Pacific': 1,
    'Europe': 2,
    'Greater China': 3,
    'India': 4,
    'Latin America': 5,
    'Middle East & Africa': 6,
    'North America': 7
}

# Loop through each region and add its data to the combined dataset
for region in df['Region']:
    # Extract the region data (ignoring 'Region' column)
    region_data = df[df['Region'] == region].iloc[:, 1:].values.flatten()
    
    # Handle missing data by replacing NaN with the average of available data for that region
    region_data = np.nan_to_num(region_data, nan=np.nanmean(region_data))
    
    # Add the region data to the combined list
    for year, value in zip(years, region_data):
        combined_data.append([region_map[region], year, value])

# Convert combined data into DataFrame
combined_df = pd.DataFrame(combined_data, columns=['Region_encoded', 'Year', 'Value'])

# Features (independent variables): 'Year' and 'Region_encoded'
X = combined_df[['Year', 'Region_encoded']]

# Target variable (dependent variable): 'Value'
y = combined_df['Value']

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate mean squared error
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.title('Actual vs Predicted Values (Linear Regression)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# Print predictions for each region and year
for region, region_id in region_map.items():
    print(f"\nPredictions for {region}:")
    for year in years:
        pred_value = model.predict([[year, region_id]])[0]
        print(f"Year {year}: {pred_value:.2f}")