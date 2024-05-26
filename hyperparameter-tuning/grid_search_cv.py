import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('wine_quality.csv')
print(df.columns)
y = df['quality']
features = df.drop(columns = ['quality'])


## 1. Data transformation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(features)

## 2. Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( X, y, random_state = 99)

## 3. Fit a logistic regression classifier without regularization
from sklearn.linear_model import LogisticRegression
clf_no_reg = LogisticRegression(penalty = "none")
clf_no_reg.fit(x_train, y_train)

## 4. Plot the coefficients
predictors = features.columns
coefficients = clf_no_reg.coef_.ravel()
coef = pd.Series(coefficients,predictors).sort_values()
coef.plot(kind='bar', title = 'Coefficients (no regularization)')
plt.tight_layout()
plt.show()
plt.clf()

## 5. Training and test performance
from sklearn.metrics import f1_score
y_pred_train = clf_no_reg.predict(x_train)
y_pred_test = clf_no_reg.predict(x_test)
f1_train = f1_score(y_train, y_pred_train)
f1_test = f1_score(y_test, y_pred_test)
print("Training score:", f1_train)
print("Testing score:", f1_test)

## 6. Default Implementation (L2-regularized!)
clf_default = LogisticRegression()
clf_default.fit(x_train, y_train)

## 7. Ridge Scores
y_pred_train = clf_default.predict(x_train)
y_pred_test = clf_default.predict(x_test)
f1_train = f1_score(y_train, y_pred_train)
f1_test = f1_score(y_test, y_pred_test)
print("Training score:", f1_train)
print("Testing score:", f1_test)

## 8. Coarse-grained hyperparameter tuning
training_array = []
test_array = []
C_array = [0.0001, 0.001, 0.01, 0.1, 1]
for x in C_array:
  clf = LogisticRegression(C = x)
  clf.fit(x_train, y_train)
  y_pred_test = clf.predict(x_test)
  y_pred_train = clf.predict(x_train)
  training_array.append(f1_score(y_train, y_pred_train))
  test_array.append(f1_score(y_test, y_pred_test))

print(C_array)
print(training_array)
print(test_array)

## 9. Plot training and test scores as a function of C
plt.plot(C_array, training_array)
plt.plot(C_array, test_array)
plt.xscale("log")
plt.show()
plt.clf()

## 10. Making a parameter grid for GridSearchCV
tuning_C = {"C": np.logspace(-4, -2, 100)}

## 11. Implementing GridSearchCV with l2 penalty
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(LogisticRegression(), param_grid = tuning_C, scoring = "f1", cv = 5)
gs.fit(x_train, y_train)

## 12. Optimal C value and the score corresponding to it
print("Optimal C:", gs.best_params_)
print("Best score:", gs.best_score_)

## 13. Validating the "best classifier"
clf_best_ridge = LogisticRegression(C = gs.best_params_["C"])
clf_best_ridge.fit(x_train, y_train)
y_pred_test = clf_best_ridge.predict(x_test)
f1 = f1_score(y_test, y_pred_test)
print("F1 score:", f1)

## 14. Implement L1 hyperparameter tuning with LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV
clf_l1 = LogisticRegressionCV(Cs = [0.01, 0.1, 1, 10, 100], cv = 5, penalty = "l1", solver = "liblinear", scoring = "f1")
clf_l1.fit(X, y)

## 15. Optimal C value and corresponding coefficients
print("Optimal C:", clf_l1.C_)
print("Optimal coef_", clf_l1.coef_)


## 16. Plotting the tuned L1 coefficients
coefficients = clf_l1.coef_.ravel()
coef = pd.Series(coefficients, predictors).sort_values()

plt.figure(figsize = (12, 8))
coef.plot(kind = "bar", title = "Coefficients for tuned L1")
plt.tight_layout()
plt.show()
plt.clf()