import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load Data from CSV
data = pd.read_csv('house_prices.csv')  # Load the new CSV file

# Step 2: Extract Features (X) and Target (y)
X = data['Square_Feet'].values.reshape(-1, 1)  # Feature: Square_Feet
y = data['Price'].values  # Target: Price

# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Step 7: Visualize the Results
plt.scatter(X, y, color='blue')  # Actual data points
plt.plot(X, model.predict(X), color='red')  # Predicted regression line
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('Linear Regression: House Price Prediction')
plt.show()
