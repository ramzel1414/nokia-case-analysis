import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define the data
age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
age_midpoints = [21, 29.5, 39.5, 49.5, 60]  # Midpoints for each age range
churn_percent = [4, 8, 17, 20, 51]  # Churn percentages

# Create a DataFrame
data = pd.DataFrame({'Age_Group': age_groups, 'Age_Midpoint': age_midpoints, 'Churn_Percent': churn_percent})

# Reshape the data for scikit-learn
X = np.array(data['Age_Midpoint']).reshape(-1, 1)  # Age midpoints as the independent variable
y = np.array(data['Churn_Percent'])  # Churn percentages as the dependent variable

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict churn percentages using the model
predictions = model.predict(X)

# Plotting the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.xlabel('Age Midpoint')
plt.ylabel('Churn Percent')
plt.title('Linear Regression of Age Groups vs. Churn Percent')
plt.legend()
plt.show()

# Output the model's coefficient and intercept
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
