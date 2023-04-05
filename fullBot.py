import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import alpaca_trade_api as tradeapi
import configparser
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Get the full path to the config file
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')

# Read the API keys from the configuration file
config = configparser.ConfigParser()
config.read(config_path)
alpaca_api_key_id = config.get('ALPACA', 'APCA_API_KEY_ID')
alpaca_secret_key = config.get('ALPACA', 'APCA_SECRET_KEY')

# Create the Alpaca API client
api = tradeapi.REST(alpaca_api_key_id, alpaca_secret_key, api_version='v2')

# Initialize the Alpaca API
# api = REST(api_key, api_secret, base_url, api_version='v2')

# Define the stock symbol and date range for the past 3 months
symbol = 'MSFT'
end_date = datetime.now() - timedelta(days=7)
start_date = end_date - timedelta(days=120)

# Fetch the stock data
df = api.get_bars(symbol, '1D', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), limit=1000).df

# Clean the data by removing open, high, and low columns
df = df.drop(columns=['open', 'high', 'low'])

stock_data = df

# Calculate the target variable (next week's close price)
df['target'] = df['close'].shift(-10)

# Drop the last 5 rows of the target variable (y) because they don't have corresponding target values
df = df[:-10]

# Split the data into a training set (80%) and a testing set (20%)
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the decision tree model
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = tree.predict(X_train_scaled)
y_test_pred = tree.predict(X_test_scaled)

# Calculate the R^2 score for the training and test sets
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Training R^2: {r2_train:.4f}')
print(f'Test R^2: {r2_test:.4f}')


# Train the decision tree
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)

# Prepare the input data for prediction (excluding 'target' column)
prediction_data = df[-11:-1].drop(columns=['target'])

# Predict the next week of trades
next_week_trades = tree.predict(prediction_data)

# Get the actual closing prices for the next week
actual_prices = df['close'][-10:].values

# Display the results using matplotlib
plt.plot(range(1, 11), next_week_trades, label='Predicted Close Price')
plt.plot(range(1, 11), actual_prices, label='Actual Close Price', linestyle='--')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.title('Predicted vs Actual Microsoft Stock Prices for Next Week')
plt.legend()
plt.show()
