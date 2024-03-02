import requests
import json
import pandas as pd

# Define API endpoint and parameters
api_key = ""
symbol = "MSFT"
interval = "1day"
output_size = "full"
start_date = pd.Timestamp.today() - pd.Timedelta(days=90)
end_date = pd.Timestamp.today()

url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key}&start_date={start_date.date()}&end_date={end_date.date()}"
# Send API request
response = requests.get(url)

# Parse JSON data
data = json.loads(response.text)

# print(data)

# Convert data to Pandas DataFrame
df = pd.DataFrame.from_dict(data["values"])

# Convert timestamp to datetime
df["datetime"] = pd.to_datetime(df["datetime"])

# Set datetime as index
df.set_index("datetime", inplace=True)
df = df.reset_index().rename(columns={'index': 'timestamp'})

# Print DataFrame
print(df)
