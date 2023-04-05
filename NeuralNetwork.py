import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Read in the cleaned stock data
data = pd.read_csv('stock_data.csv')

# Drop the timestamp column
data = data.drop(columns=['timestamp'])

# Normalize the data
scaler = MinMaxScaler()
df_norm = scaler.fit_transform(data.values)

# Create a separate MinMaxScaler for the target variable
target_scaler = MinMaxScaler()
target_scaler.fit(data.iloc[:, -1].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(df_norm) * 0.8)
train_data = df_norm[:train_size]
test_data = df_norm[train_size:]

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

# Define the hyperparameters
input_size = train_data.shape[1] - 1
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 100

# Train the neural network
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    inputs = torch.autograd.Variable(torch.Tensor(train_data[:, :-1]).float())
    labels = torch.autograd.Variable(torch.Tensor(train_data[:, -1]).float())

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs.squeeze(), labels)
    loss.backward()
    optimizer.step()

    print("Epoch {0}, loss: {1}".format(epoch+1, loss.item()))

# Test the neural network
model.eval()
test_inputs = torch.autograd.Variable(torch.Tensor(test_data[:, :-1]).float())
test_labels = torch.autograd.Variable(torch.Tensor(test_data[:, -1]).float())
test_outputs = model(test_inputs).squeeze().detach().numpy()

# Invert the normalization
test_labels = target_scaler.inverse_transform(test_labels.reshape(-1, 1)).flatten()
test_outputs = target_scaler.inverse_transform(test_outputs.reshape(-1, 1)).flatten()

# Evaluate the neural network
mse = mean_squared_error(test_labels, test_outputs)
r2 = r2_score(test_labels, test_outputs)
print("Neural network MSE: {0}".format(mse))
print("Neural network R^2 score: {0}".format(r2))


# # Split the data into training and testing sets
# X = data.drop('AAPL_close', axis=1).values
# y = data['AAPL_close'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Define the neural network architecture
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(X_train.shape[1], 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 1)

#     def forward(self, x):
#         x = nn.functional.relu(self.fc1(x))
#         x = nn.functional.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Initialize the neural network and set the loss function and optimizer
# net = Net()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)

# # Train the neural network
# for epoch in range(100):
#     running_loss = 0.0
#     for i, data in enumerate(X_train):
#         inputs = torch.tensor(X_train[i]).float()
#         labels = torch.tensor(y_train[i]).float()

#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"Epoch {epoch + 1}, loss: {running_loss / len(X_train)}")

# # Evaluate the neural network
# y_pred = []
# with torch.no_grad():
#     for i, data in enumerate(X_test):
#         inputs = torch.tensor(X_test[i]).float()
#         outputs = net(inputs)
#         y_pred.append(outputs.item())

# mse = np.mean((y_test - y_pred) ** 2)
# print(f"Neural Network MSE: {mse}")
