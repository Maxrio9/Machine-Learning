import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names = cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate']

#Print number of countries by landmass, or continent
print(df.groupby("landmass").name.count().reset_index())

#Create a new dataframe with only flags from Europe and Oceania
df_36 = df[df["landmass"].isin([3, 6])]
print("Len of df_36:", len(df_36))

#Print the average vales of the predictors for Europe and Oceania
print(df_36.groupby("landmass")[var].mean())

#Create labels for only Europe and Oceania
labels = df_36["landmass"]

#Print the variable types for the predictors
print(df_36[var].dtypes)

#Create dummy variables for categorical predictors
data = pd.get_dummies(df_36[var], drop_first = True)

#Split data into a train and test set
x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state = 1, test_size = 0.4)

#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []
for depth in depths:
  dt = DecisionTreeClassifier(max_depth = depth)
  dt.fit(x_train, y_train)
  acc = dt.score(x_test, y_test)
  acc_depth.append(acc)

#Plot the accuracy vs depth
plt.plot(depths, acc_depth)
plt.xlabel("Depth of Search")
plt.ylabel("Accuracy")
plt.title("Accuracy for varying depth")
plt.show()
plt.clf()

#Find the largest accuracy and the depth this occurs
lar_acc = max(acc_depth)
lar_acc_index = acc_depth.index(lar_acc) + 1
print(lar_acc_index, lar_acc)

#Refit decision tree model with the highest accuracy and plot the decision tree
dt = DecisionTreeClassifier(max_depth = lar_acc_index)
dt.fit(x_train, y_train)
tree.plot_tree(dt)
plt.show()
plt.clf()

#Create a new list for the accuracy values of a pruned decision tree.  Loop through
#the values of ccp and append the scores to the list
ccp_alpha = [i / 100.0 for i in range(11)]
acc_pruned = []
for ccp in ccp_alpha:
  dt = DecisionTreeClassifier(max_depth = lar_acc_index, ccp_alpha = ccp)
  dt.fit(x_train, y_train)
  score = dt.score(x_test, y_test)
  acc_pruned.append(score)

#Plot the accuracy vs ccp_alpha
plt.plot(ccp_alpha, acc_pruned)
plt.title("Accuracy for different ccp values")
plt.xlabel("CCP")
plt.ylabel("Accuracy")
plt.show()
plt.clf()

#Find the largest accuracy and the ccp value this occurs
lar_ccp_acc = max(acc_pruned)
best_ccp = acc_pruned.index(lar_ccp_acc) / 100.0
print(best_ccp, lar_ccp_acc)

#Fit a decision tree model with the values for max_depth and ccp_alpha found above
dt = DecisionTreeClassifier(max_depth = lar_acc_index, ccp_alpha = best_ccp)
dt.fit(x_train, y_train)

#Plot the final decision tree
tree.plot_tree(dt)
plt.show()
plt.clf()
