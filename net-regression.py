import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data: Year and corresponding net worth values (in billions)
data = {
    'Year': [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'Net_Worth': [19.77, 30.38, 31.19, 30.02, 29.53, 29.37, 34.19, 41.12, 51.06, 50.71, 40.98, 42.45, 38.66, 30.18, 12.71, 11.76, 12.5, 23.64, 23.15, 22.56, 23.22, 21.85, 22.2, 24.9, 22.26]
}

# Converting the data into a pandas DataFrame
df = pd.DataFrame(data)

# Reshaping the data for the regression model
X = df['Year'].values.reshape(-1, 1)  # Features (Year)
y = df['Net_Worth'].values  # Target (Net worth)

# Creating and fitting the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Making predictions
y_pred = model.predict(X)

# Model evaluation
mse = mean_squared_error(y, y_pred)  # Mean Squared Error
r2 = r2_score(y, y_pred)  # R-squared value

# Print the results
print(f"Linear Regression Model Coefficient: {model.coef_[0]}")
print(f"Linear Regression Model Intercept: {model.intercept_}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plotting the data and the regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title('Nokia Net Worth (1991-2023)')
plt.xlabel('Year')
plt.ylabel('Net Worth (in billions)')
plt.legend()
plt.show()