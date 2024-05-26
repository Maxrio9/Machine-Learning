import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Load the data set
cancer = load_breast_cancer()

# Split the data into training and testing sets
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create distributions to draw hyperparameters from
distributions = {'penalty': ['l1', 'l2'], 'C': uniform(loc=0, scale=100)}

# The logistic regression model
lr = LogisticRegression(solver = 'liblinear', max_iter = 1000)

# Create a RandomizedSearchCV model
clf = RandomizedSearchCV(lr, distributions, n_iter=8)


clf.fit(X_train, y_train)
best_model = clf.best_estimator_

print(best_model)
print(clf.best_params_)


best_score = clf.best_score_
test_score = clf.score(X_test, y_test)

print(best_score)
print(test_score)


hyperparameter_values = pd.DataFrame(clf.cv_results_["params"])
randomsearch_scores = pd.DataFrame(clf.cv_results_["mean_test_score"], columns = ["score"])

df = pd.concat([hyperparameter_values, randomsearch_scores], axis = 1)
print(df)