import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import alpaca_trade_api as tradeapi
import configparser

# Get the full path to the config file
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')

# Read the API keys from the configuration file
config = configparser.ConfigParser()
config.read(config_path)
alpaca_api_key_id = config.get('ALPACA', 'APCA_API_KEY_ID')
alpaca_secret_key = config.get('ALPACA', 'APCA_SECRET_KEY')

# Create the Alpaca API client
api = tradeapi.REST(alpaca_api_key_id, alpaca_secret_key, api_version='v2')

# Define the stock symbol and date range for the past 3 months
symbol = 'MSFT'
end_date = datetime.now() - timedelta(days=7)
start_date = end_date - timedelta(days=120)

# Fetch the stock data
df = api.get_bars(symbol, '1D', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), limit=1000).df

# Clean the data by removing open, high, and low columns
df = df.drop(columns=['open', 'high', 'low'])

# Calculate the target variable (next week's close price)
df['target'] = df['close'].shift(-5)

# Drop the last 5 rows of the target variable (y) because they don't have corresponding target values
df = df[:-5]

# Split the data into a training set (80%) and a testing set (20%)
X = df.drop(columns=['target']).values
y = df['target'].dropna().values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(df)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Prepare the data for PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).view(-1, 1, 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).view(-1, 1, 1)

# Define the GRU model
class gru_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(gru_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out



# Set the hyperparameters
input_size = X_train_tensor.shape[2]
hidden_size = 64
num_layers = 1
output_size = 1
num_epochs = 100
learning_rate = 0.001

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, loss function, and optimizer
model = gru_model(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the GRU model
num_epochs = 50
learning_rate = 0.001

# Prepare the train_input and train_target tensors
train_input = torch.from_numpy(X_train_scaled).view(-1, 1, 1).float().to(device)
train_target = torch.from_numpy(y_train).float().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(train_input)
    loss = criterion(output, train_target)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions
test_input = torch.from_numpy(X_test_scaled).view(-1, 1, 1).float().to(device)
predicted_stock_prices = model(test_input).detach().cpu().numpy()

# Calculate the mean squared error
mse = np.mean((predicted_stock_prices - y_test)**2)
print(f'Mean Squared Error: {mse:.4f}')

# Get the actual closing prices for the next week
actual_prices = y_test[-5:]

# Display the results using matplotlib
plt.plot(range(1, 6), predicted_stock_prices[-5:], label='Predicted Close Price')
plt.plot(range(1, 6), actual_prices, label='Actual Close Price', linestyle='--')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.title('Predicted vs Actual Microsoft Stock Prices for Next Week')
plt.legend()
plt.show()