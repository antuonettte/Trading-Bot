import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import configparser
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import time
import websocket
import json
from pprint import pprint

# Get the full path to the config file
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')

# Read the API keys from the configuration file
config = configparser.ConfigParser()
config.read(config_path)
alpaca_api_key_id = config.get('ALPACA', 'APCA_API_KEY_ID')
alpaca_secret_key = config.get('ALPACA', 'APCA_SECRET_KEY')
# Set the base URL for paper trading
PAPER_BASE_URL = "https://paper-api.alpaca.markets"
SOCKET_URL = "wss://paper-api.alpaca.markets/stream"

# Create the Alpaca API client
api = tradeapi.REST(alpaca_api_key_id, alpaca_secret_key, api_version='v2')

# Create the Alpaca API client for paper trading
paperApi = tradeapi.REST(alpaca_api_key_id, alpaca_secret_key, base_url=PAPER_BASE_URL, api_version="v2")

# Define the stock symbol
symbol = 'MSFT'

# Initialize the model
model = DecisionTreeRegressor()

window_size = 30
prediction_horizon = 1

FIXED_AMOUNT_SHARES = 10

trade_reset_time = None
trades_this_week = 0

# WebSocket functions
def on_open(ws):
    print("WebSocket opened")
    auth_data = {
        "action": "authenticate",
        "data": {
            "key_id": alpaca_api_key_id,
            "secret_key": alpaca_secret_key
        }
    }
    ws.send(json.dumps(auth_data))

    listen_data = {
        "action": "listen",
        "data": {
            "streams": ["trade_updates"]
        }
    }
    ws.send(json.dumps(listen_data))


def on_message(ws, message):
    global trade_reset_time
    global trades_this_week
    global account

    # Deserialize the message into a JSON object
    message_json = json.loads(message)

    # Check if the message is a trade update
    if 'T' in message_json:
        # Extract the stock symbol from the message
        symbol = message_json['S']

        # Run the trading algorithm and make predictions
        # Update the end_date and start_date to get the latest data
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=120)

        # Fetch the latest stock data
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

        # Prepare the input data for prediction (excluding 'target' column)
        prediction_data = df[-1:].drop(columns=['target'])

        # Predict the next day of trades
        next_day_trades = model.predict(prediction_data)

        # Calculate the predicted price change ratio
        predicted_price_change_ratio = next_day_trades[-1] / df['close'].iloc[-1]

        # Get the account information
        account = paperApi.get_account()

        # Buy Logic
        if predicted_price_change_ratio > 1.01:
            # Calculate the maximum number of shares to purchase based on the account's available buying power
            max_shares_to_buy = int(float(account.buying_power) / df['close'].iloc[-1])

            # Calculate the number of shares to purchase (
        # Check if the trade reset time is set and if the current time is past the reset time
        if trade_reset_time is not None and datetime.now() > trade_reset_time:
            trades_this_week = 0
            trade_reset_time = None

        # If there are fewer than 3 trades this week, continue with trading
        if trades_this_week < 3:
            # Place a market order to buy the shares
            if shares_to_buy > 0:
                paperApi.submit_order(
                    symbol=symbol,
                    qty=shares_to_buy,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Bought {shares_to_buy} shares of {symbol}")
                trades_this_week += 1

    # Sell Logic
    elif predicted_price_change_ratio < 0.99:
        try:
            position = paperApi.get_position(symbol)
            if float(position.qty) > 0:  # Check if we have a long position
                # Place a market order to sell all the shares
                paperApi.submit_order(
                    symbol=symbol,
                    qty=position.qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Sold {position.qty} shares of {symbol}")
                trades_this_week += 1
        except Exception as e:
            print(f"Error placing sell order for {symbol}: {str(e)}")

    # If we executed a trade today, set the trade reset time to the next Monday (assuming the market is closed on weekends)
    if trades_this_week > 0:
        trade_reset_time = datetime.now() + timedelta(days=(7 - datetime.now().weekday()))
        print(f"Trade reset time set to: {trade_reset_time}")


def on_error(ws, error):
    print(f"Error: {error}")


def on_close(ws):
    print("WebSocket closed")


# Initialize the WebSocket
ws = websocket.WebSocketApp(SOCKET_URL, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
ws.run_forever()

# while True:
#     # Update the end_date and start_date to get the latest data
#     end_date = datetime.now() - timedelta(days=1)
#     start_date = end_date - timedelta(days=120)

#     # Fetch the latest stock data
#     df = api.get_bars(symbol, '1D', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), limit=1000).df

#     # Clean the data by removing open, high, and low columns and convert time to string
#     df = df.drop(columns=['open', 'high', 'low'])
#     df = df.reset_index().rename(columns={'index': 'timestamp'})
#     df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date.apply(lambda x: int(x.strftime('%Y%m%d')))

#     window_size = 30
#     prediction_horizon = 10

#     # Calculate the target variable (next week's close price) for the entire dataframe
#     df['target'] = df['close'].shift(-prediction_horizon)

#     # Initialize the model
#     model = DecisionTreeRegressor()

#     # Create empty lists to store the predictions and actual values
#     predictions = []
#     actual_values = []

#     # Iterate through the time series using a rolling window
#     for i in range(len(df) - window_size - prediction_horizon + 1):
#         # Extract the data within the window
#         window_data = df.iloc[i:i + window_size]

#         # Split the data into features (X) and target (y)
#         X = window_data.drop(columns=['target'])
#         y = window_data['target'].dropna()  # Drop any NaN values

#         # Train the model on the data within the window
#         model.fit(X, y)

#         # Make a prediction for the next data point outside the window
#         next_data_point = df.iloc[i + window_size].drop(labels=['target'])
#         prediction = model.predict(np.array(next_data_point).reshape(1, -1))

#         # Store the prediction and the actual value
#         predictions.append(prediction[0])
#         actual_values.append(df.iloc[i + window_size + prediction_horizon - 1]['close'])

#     # Calculate the R^2 score
#     r2 = r2_score(actual_values, predictions)
#     print(f'R^2 Score: {r2:.4f}')

#     # Prepare the input data for prediction (excluding 'target' column)
#     prediction_data = df[-1:].drop(columns=['target'])

#     # Predict the next day of trades
#     next_day_trades = model.predict(prediction_data)

#     # Calculate the predicted price change ratio
#     predicted_price_change_ratio = next_day_trades[-1] / df['close'].iloc[-1]

#     # Get the account information
#     account = paperApi.get_account()

#     # Check if the predicted price change ratio is greater than 1.01, indicating a potential increase in price
#     if predicted_price_change_ratio > 1.01:
#         # Calculate the maximum number of shares to purchase based on the account's available buying power
#         max_shares_to_buy = int(float(account.buying_power) / df['close'].iloc[-1])

#         # Calculate the number of shares to purchase (limited to the maximum number of shares allowed)
#         shares_to_buy = min(max_shares_to_buy, FIXED_AMOUNT_SHARES)

#         # Check if the trade reset time is set and if the current time is past the reset time
#         if trade_reset_time is not None and datetime.now() > trade_reset_time:
#             trades_this_week = 0
#             trade_reset_time = None

#         # If there are fewer than 3 trades this week, continue with trading
#         if trades_this_week < 3:
#             # Place a market order to buy the shares
#             if shares_to_buy > 0:
#                 paperApi.submit_order(
#                     symbol=symbol,
#                     qty=shares_to_buy,
#                     side='buy',
#                     type='market',
#                     time_in_force='gtc'
#                 )
#                 print(f"Bought {shares_to_buy} shares of {symbol}")

#     # If we have a position in the stock, check if the predicted price change ratio is less than 0.99, indicating a potential decrease in price
#     elif predicted_price_change_ratio < 0.99:
#         try:
#             position = paperApi.get_position(symbol)
#             if float(position.qty) > 0:  # Check if we have a long position
#                 # Place a market order to sell all the shares
#                 paperApi.submit_order(
#                     symbol=symbol,
#                     qty=position.qty,
#                     side='sell',
#                     type='market',
#                     time_in_force='gtc'
#                 )
#                 print(f"Sold {position.qty} shares of {symbol}")
#         except Exception as e:
#             print(f"Error placing sell order for {symbol}: {str(e)}")

    # Wait for one day before checking the predictions again
    time.sleep(60 * 60 * 24)  # Wait for 24 hours


