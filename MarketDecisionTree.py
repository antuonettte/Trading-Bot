import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib


# Load the data from Alpaca API or a CSV file
# data = ... (your code to get data from Alpaca API or load from CSV)
data = pd.read_csv('stock_data.csv')


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=[ 'AAPL_close']), data['AAPL_close'], test_size=0.2)

# Train the decision tree model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Evaluate the model on the testing set
score = model.score(X_test, y_test)

joblib.dump(model, 'decision_tree_model.sav')


print(f"Decision Tree R^2 Score: {score}")