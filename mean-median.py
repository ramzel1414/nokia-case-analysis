import numpy as np
import matplotlib.pyplot as plt

# Data: Years and corresponding sales values
years = np.array([
    1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
    2019, 2020, 2021, 2022, 2023
])
sales = np.array([
    19.77, 30.38, 31.19, 30.02, 29.53, 29.37, 34.19, 41.12, 51.06, 50.71,
    40.98, 42.45, 38.66, 30.18, 12.71, 11.76, 12.50, 23.64, 23.15, 22.56,
    23.32, 21.85, 22.20, 24.90, 22.26
])

# Calculate the mean and median of the sales
mean_sales = np.mean(sales)
median_sales = np.median(sales)

# Bar graph of mean and median sales
labels = ['Mean', 'Median']
values = [mean_sales, median_sales]

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(labels, values, color=['#A3B7D4', '#A3D7A6'])

# Add labels and title
plt.ylabel('Net Sales (in Billion Euros)')
plt.title('Nokia\'s Net Sales: Mean vs Median (1999-2023)')

# Display the plot
plt.show()

# Output the mean and median values
print(f"Mean of Sales: {mean_sales:.2f} Billion Euros")
print(f"Median of Sales: {median_sales:.2f} Billion Euros")
