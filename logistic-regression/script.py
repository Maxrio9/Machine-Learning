import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
print(transactions.info())

# How many fraudulent transactions?
print(transactions.isFraud.value_counts()[1])

# Summary statistics on amount column
print(transactions.amount.describe())
# Very Heavily skewed to the right

# Create isPayment field
transactions["isPayment"] = transactions.type.apply(lambda x: 1 if x == "PAYMENT" or x == "DEBIT" else 0)

# Create isMovement field
transactions["isMovement"] = transactions.type.apply(lambda x: 1 if x == "CASH_OUT" or x == "TRANSFER" else 0)

# Create accountDiff field
transactions["accountDiff"] = abs(transactions.oldbalanceOrg - transactions.oldbalanceDest)

# Create features and label variables
features = transactions[["amount", "isPayment", "isMovement", "accountDiff"]]

label = transactions[["isFraud"]]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.3)

# Normalize the features variables
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Fit the model to the training data
cc_lr = LogisticRegression()
cc_lr.fit(x_train, y_train)

# Score the model on the training data
print(cc_lr.score(x_train, y_train))

# Score the model on the test data
print(cc_lr.score(x_test, y_test))

# Print the model coefficients
print(cc_lr.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
transaction4 = np.array([1234456.78, 1.0, 1.0, 566670.1])
transaction5 = np.array([18765.43, 0.0, 1.0, 18524.75])
transaction6 = np.array([2678.31, 0.0, 0.0, 200.5])

# Combine new transactions into a single array
sample_transactions = np.array([transaction1, transaction2, transaction3, transaction4, transaction5, transaction6])

# Normalize the new transactions
sample_transactions = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
print(cc_lr.predict(sample_transactions))

# Show probabilities on the new transactions
print(cc_lr.predict_proba(sample_transactions))