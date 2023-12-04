import os
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# Read data
file_path = os.path.join("/kaggle/input/aq-dataset", "dataset_aq 2 - SAS input.csv")
df = pd.read_csv(file_path, encoding='latin1')

# Handle missing values
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df = df.dropna()

# Perform linear regression
model = ols('Value ~ Year + PPM_uom + C(Pollutant) + C(Country)', data=df).fit()

# Print summary for reference
print(model.summary())

future_years = pd.DataFrame({
    'Year': range(2023, 2051),
    'PPM_uom': np.random.rand(28),  
    'Pollutant': np.random.choice(['pm25', 'pm10', 'so2', 'o3', 'no2'], 28),
    'Country' : np.random.choice(['US', 'CA'], 28)
})

predicted_values = model.predict(future_years)
future_years['Predicted_Value_Of_Pollutant'] = predicted_values
print(future_years)

# Data visualization
plt.figure(figsize=(10, 6))

# Scatter plot of actual data
plt.scatter(df['Year'], df['Value'], label='Actual Data', color='blue')

# Plot the regression line
plt.plot(future_years['Year'], future_years['Predicted_Value_Of_Pollutant'], label='Linear Regression', color='red')

# Labeling
plt.title('Linear Regression of Pollution over Time')
plt.xlabel('Year')
plt.ylabel('Pollution Value')
plt.legend()


plt.show()
