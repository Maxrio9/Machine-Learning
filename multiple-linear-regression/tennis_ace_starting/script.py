import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# load and investigate the data here:
tennis = pd.read_csv("tennis_stats.csv")
print(tennis.head())
print(tennis.describe())
print(tennis.columns)

# perform exploratory analysis here:
# sns.pairplot(tennis.dropna(), kind = "reg")


## perform single feature linear regressions here:
x_train, x_test, y_train, y_test = train_test_split(tennis[["BreakPointsOpportunities"]], tennis["Winnings"], train_size = 0.8, test_size = 0.2)

lrg1 = LinearRegression()

lrg1.fit(x_train, y_train)
y_predict = lrg1.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.4)

plt.show()
plt.clf()

## perform two feature linear regressions here:

x_train, x_test, y_train, y_test = train_test_split(tennis[["BreakPointsOpportunities", "FirstServeReturnPointsWon"]], tennis["Winnings"], train_size = 0.8, test_size = 0.2)

lrg2 = LinearRegression()

lrg2.fit(x_train, y_train)
y_predict = lrg2.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.4)

plt.show()
plt.clf()


## perform multiple feature linear regressions here:

x = tennis[["FirstServe", "FirstServePointsWon", "FirstServeReturnPointsWon", "SecondServePointsWon", "SecondServeReturnPointsWon", "Aces", "BreakPointsConverted", "BreakPointsFaced", "BreakPointsOpportunities", "BreakPointsSaved", "DoubleFaults", "ReturnGamesPlayed", "ReturnGamesWon", "ReturnPointsWon", "ServiceGamesPlayed", "ServiceGamesWon", "TotalPointsWon", "TotalServicePointsWon"]]
y = tennis[["Winnings"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)

lrg2 = LinearRegression()

lrg2.fit(x_train, y_train)
y_predict = lrg2.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.4)
plt.title("Actual vs Predicted Earnings")
plt.xlabel("Actual Earnings")
plt.ylabel("Predicted Earnings")
plt.show()
plt.clf()

# We can see the multiple linear regression models fits really well to the average but outliers are neglected