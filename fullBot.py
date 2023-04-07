import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import alpaca_trade_api as tradeapi
import configparser
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def place_order(symbol, qty, side, order_type, time_in_force):
    try:
        paperApi.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )
        print(f"{side.capitalize()} order placed for {symbol}: {qty} shares")
    except Exception as e:
        print(f"Error placing {side} order for {symbol}: {str(e)}")

# Get the full path to the config file
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')

# Read the API keys from the configuration file
config = configparser.ConfigParser()
config.read(config_path)
alpaca_api_key_id = config.get('ALPACA', 'APCA_API_KEY_ID')
alpaca_secret_key = config.get('ALPACA', 'APCA_SECRET_KEY')

# Set the base URL for paper trading
BASE_URL = "https://paper-api.alpaca.markets"

# Create the Alpaca API client
paperApi = tradeapi.REST(alpaca_api_key_id, alpaca_secret_key, base_url=BASE_URL, api_version="v2")

# Create the Alpaca API client
api = tradeapi.REST(alpaca_api_key_id, alpaca_secret_key, api_version='v2')

# Initialize the Alpaca API
# api = REST(api_key, api_secret, base_url, api_version='v2')

# Define the stock symbol and date range for the past 3 months
symbol = 'AAPL'
end_date = datetime.now() - timedelta(days=7)
start_date = end_date - timedelta(days=120)

# Fetch the stock data
df = api.get_bars(symbol, '1D', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), limit=1000).df

# Clean the data by removing open, high, and low columns and convert time to string
df = df.drop(columns=['open', 'high', 'low'])
df = df.reset_index().rename(columns={'index': 'timestamp'})
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date.apply(lambda x: int(x.strftime('%Y%m%d')))


window_size = 30
prediction_horizon = 10

# Calculate the target variable (next week's close price) for the entire dataframe
df['target'] = df['close'].shift(-prediction_horizon)

# Initialize the model
model = DecisionTreeRegressor()

# Create empty lists to store the predictions and actual values
predictions = []
actual_values = []

# Iterate through the time series using a rolling window
for i in range(len(df) - window_size - prediction_horizon + 1):
    # Extract the data within the window
    window_data = df.iloc[i:i + window_size]

    # Split the data into features (X) and target (y)
    X = window_data.drop(columns=['target'])
    y = window_data['target'].dropna()  # Drop any NaN values

    # Train the model on the data within the window
    model.fit(X, y)

    # Make a prediction for the next data point outside the window
    next_data_point = df.iloc[i + window_size].drop(labels=['target'])
    prediction = model.predict(np.array(next_data_point).reshape(1, -1))

    # Store the prediction and the actual value
    predictions.append(prediction[0])
    actual_values.append(df.iloc[i + window_size + prediction_horizon - 1]['close'])

# Calculate the R^2 score
r2 = r2_score(actual_values, predictions)
print(f'R^2 Score: {r2:.4f}')

# Plot the predicted vs actual closing prices
plt.plot(range(1, len(predictions) + 1), predictions, label='Predicted Close Price')
plt.plot(range(1, len(actual_values) + 1), actual_values, label='Actual Close Price', linestyle='--')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.title(f'Predicted vs Actual {symbol} Stock Prices')
plt.legend()
# plt.ion()
plt.show()
# plt.ioff()

# Get the latest trade data and make a prediction
next_data_point = df.iloc[-1].drop(labels=['target'])
predicted_price = model.predict(np.array(next_data_point).reshape(1, -1))[0]

print(predicted_price)

# Get the current stock price
current_price = api.get_latest_trade(symbol).price

print(current_price)

# Define the number of shares to trade
trade_qty = 1

# Place a buy order if the predicted price is higher than the current price
if predicted_price > current_price:
    place_order(symbol, trade_qty, "buy", "market", "gtc")

# Place a sell order if the predicted price is lower than the current price
elif predicted_price < current_price:
    place_order(symbol, trade_qty, "sell", "market", "gtc")

# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# import alpaca_trade_api as tradeapi
# import configparser
# from sklearn.metrics import r2_score
# from sklearn.preprocessing import StandardScaler

# # Get the full path to the config file
# config_path = os.path.join(os.path.dirname(__file__), 'config.ini')

# # Read the API keys from the configuration file
# config = configparser.ConfigParser()
# config.read(config_path)
# alpaca_api_key_id = config.get('ALPACA', 'APCA_API_KEY_ID')
# alpaca_secret_key = config.get('ALPACA', 'APCA_SECRET_KEY')

# # Create the Alpaca API client
# api = tradeapi.REST(alpaca_api_key_id, alpaca_secret_key, api_version='v2')

# # Initialize the Alpaca API
# # api = REST(api_key, api_secret, base_url, api_version='v2')

# # Define the stock symbol and date range for the past 3 months
# symbol = 'XOM'
# end_date = datetime.now() - timedelta(days=7)
# start_date = end_date - timedelta(days=120)

# # Fetch the stock data
# df = api.get_bars(symbol, '1D', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), limit=1000).df

# # Clean the data by removing open, high, and low columns and convert time to string
# df = df.drop(columns=['open', 'high', 'low'])
# df = df.reset_index().rename(columns={'index': 'timestamp'})
# df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date.apply(lambda x: int(x.strftime('%Y%m%d')))

# stock_data = df

# window_size = 30
# prediction_horizon = 10

# # Calculate the target variable (next week's close price) for the entire dataframe
# df['target'] = df['close'].shift(-prediction_horizon)

# # Initialize the model
# model = DecisionTreeRegressor()

# # Create empty lists to store the predictions and actual values
# predictions = []
# actual_values = []

# # Iterate through the time series using a rolling window
# for i in range(len(df) - window_size - prediction_horizon + 1):
#     # Extract the data within the window
#     window_data = df.iloc[i:i + window_size]

#     # Split the data into features (X) and target (y)
#     X = window_data.drop(columns=['target'])
#     y = window_data['target'].dropna()  # Drop any NaN values

#     # Train the model on the data within the window
#     model.fit(X, y)

#     # Make a prediction for the next data point outside the window
#     next_data_point = df.iloc[i + window_size].drop(labels=['target'])
#     prediction = model.predict(np.array(next_data_point).reshape(1, -1))

#     # Store the prediction and the actual value
#     predictions.append(prediction[0])
#     actual_values.append(df.iloc[i + window_size + prediction_horizon - 1]['close'])

# # Calculate the R^2 score
# r2 = r2_score(actual_values, predictions)
# print(f'R^2 Score: {r2:.4f}')

# # Plot the predicted vs actual closing prices
# plt.plot(range(1, len(predictions) + 1), predictions, label='Predicted Close Price')
# plt.plot(range(1, len(actual_values) + 1), actual_values, label='Actual Close Price', linestyle='--')
# plt.xlabel('Days')
# plt.ylabel('Close Price')
# plt.title(f'Predicted vs Actual {symbol} Stock Prices')
# plt.legend()
# plt.show()