import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data: Years and corresponding sales values
years = np.array([
    1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
    2019, 2020, 2021, 2022, 2023
]).reshape(-1, 1)
sales = np.array([
    19.77, 30.38, 31.19, 30.02, 29.53, 29.37, 34.19, 41.12, 51.06, 50.71,
    40.98, 42.45, 38.66, 30.18, 12.71, 11.76, 12.50, 23.64, 23.15, 22.56,
    23.32, 21.85, 22.20, 24.90, 22.26
])
# Initialize and train the linear regression model
model = LinearRegression()
model.fit(years, sales)

# Predict the sales based on the regression line
sales_pred = model.predict(years)

# Plot the data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(years, sales, color='blue', label='Actual Sales')
plt.plot(years, sales_pred, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Net Sales (in Billion Euros)')
plt.title('Linear Regression of Nokia\'s Net Sales (1999-2023)')
plt.legend()

# Display the plot
plt.show()

# Output the slope and intercept of the regression line
slope = model.coef_[0]
intercept = model.intercept_
