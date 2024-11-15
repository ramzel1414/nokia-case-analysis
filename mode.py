import pandas as pd
import matplotlib.pyplot as plt

# Data for years 2018-2023 and their respective values for customer categories
customer_data = {
    'Year': [2018, 2019, 2020, 2021, 2022, 2023],
    'Communication Service Providers': [1167, 1409, 1502, 1575, 2007, 2282],
    'Enterprise': [449, 447, 440, 440, 440, 440],
    'Licenses': [1148, 1391, 1571, 1575, 2007, 2282],
    'Other': [925, 1148, 1402, 1571, 1502, 1575]
}

# Convert customer data into a DataFrame
df_customer = pd.DataFrame(customer_data)

# Calculate the mode for each customer category
# If there is no mode, assume the middle value (median) as the mode
modes = df_customer.drop('Year', axis=1).mode().iloc[0]

# Handling cases where mode doesn't exist: assume median value
for category in modes.index:
    if modes[category] == pd.Series(df_customer[category]).mode().max():  # If mode is NaN, assume median
        modes[category] = df_customer[category].median()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(modes, labels=modes.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
plt.title('Mode of Customer Categories (2018-2023) with Assumed Middle Values for Missing Modes')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular
plt.show()
