import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved decision tree model
model = joblib.load('decision_tree_model.sav')

# Load the test data for the next week
test_data = pd.read_csv('stock_data.csv')

# Drop timestamp and AAPL_close columns
test_data.drop(columns=['timestamp', 'AAPL_close'], inplace=True)

print(test_data)

# Make predictions using the decision tree model
y_pred = model.predict(test_data)

# print(y_pred)

# Plot a graph of the predicted values against the actual values
plt.plot(y_pred, label='Predicted')
plt.plot(test_data['XOM_close'], label='Actual')
plt.legend()
plt.show()
