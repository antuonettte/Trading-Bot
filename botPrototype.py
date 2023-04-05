import matplotlib.pyplot as plt
import configparser
import alpaca_trade_api as tradeapi
import os
from datetime import datetime, timedelta
import pandas as pd
from pprint import pprint

# Get the full path to the config file
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')

# Read the API keys from the configuration file
config = configparser.ConfigParser()
config.read(config_path)
alpaca_api_key_id = config.get('ALPACA', 'APCA_API_KEY_ID')
alpaca_secret_key = config.get('ALPACA', 'APCA_SECRET_KEY')

# Create the Alpaca API client
api = tradeapi.REST(alpaca_api_key_id, alpaca_secret_key, api_version='v2')

# define the stocks we want to analyze
stocks = ['AAPL', 'MSFT', 'XOM', 'AMD']

end = datetime.now() - timedelta(days=7)
start = end - timedelta(days=120)

# get the stock data for each stock
data = {}
for stock in stocks:
    data[stock] = api.get_bars(stock, '1D', start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), limit=500).df




# Create a Pandas DataFrame with the data
df = pd.concat(data.values(), axis=1, keys=['AAPL', 'MSFT', 'XOM', 'AMD'], names=['Symbol', 'Metrics'])

new_columns = [f'{symbol}_{metric}' for symbol, metric in df.columns]

df.columns = new_columns

df = df.loc[:,~df.columns.duplicated()]

df.dropna(inplace=True)

# Drop unwanted columns
df.drop(['AAPL_trade_count', 'MSFT_trade_count', 'XOM_trade_count', 'AMD_trade_count'], axis=1, inplace=True)
df.drop(['AAPL_high', 'MSFT_high', 'XOM_high', 'AMD_high'], axis=1, inplace=True)
df.drop(['AAPL_low', 'MSFT_low', 'XOM_low', 'AMD_low'], axis=1, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Create the target column by shifting the 'AAPL_close' column up by one row
df['target'] = df['AAPL_close'].shift(-1)

# Drop the last row since it has a NaN value in the target column
df.drop(df.tail(1).index, inplace=True)

print(df.index.tolist())

# Convert the timestamp format
# data['timestamp'] = pd.to_datetime(data['timestamp']).apply(lambda x: x.isoformat())

# Write the data to a CSV file
df.to_csv('stock_data.csv')

# plot the closing prices for each stock
for stock in stocks:
    data[stock]['close'].plot()

# add labels and a legend
plt.title('Closing Prices for Selected Stocks')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(stocks)

# show the plot
plt.show()