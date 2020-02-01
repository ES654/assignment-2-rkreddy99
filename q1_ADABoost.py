"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from linearRegression.linearRegression import LinearRegression


np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################
print("-----------------------------------------------------------")
print("Predicting on random data")
print("-----------------------------------------------------------")
N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
# [fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features
print("-----------------------------------------------------------")
print("Predicting on iris data")
print("-----------------------------------------------------------")

X = pd.read_csv("iris.data")
# 
a = []
for i in range(5):
  if i==4:
    a.append(X.columns[i])
  else:
    a.append(float(X.columns[i]))
col = ["sepal length", "sepal width", "petal length", "petal width", "label"]
X.columns = col
d={}
for i in range(5):
  d[col[i]] = a[i]
X = X.append(d, ignore_index=True)

X = X.drop(["sepal length"], axis=1)
X = X.drop(["petal length"], axis=1)


for i in range(X["label"].size):
    if X["label"][i]!="Iris-virginica":
        X["label"][i] = "Iris-non-virginica"

ind = [i for i in range(X.shape[0])]
np.random.shuffle(ind)
Xtemp = X
for i in range(len(ind)):
    X.loc[i] = Xtemp.loc[ind[i]]
X = X.reset_index(drop=True)
y = X["label"]

X_train = X.loc[:89].reset_index(drop=True)
y_train = X_train["label"]
X_train = X_train.drop(["label"], axis=1)

X_test = X[90:].reset_index(drop=True)
y_test = X_test["label"]
X_train = X_test.drop(["label"], axis=1)

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X_train, y_train)
y_hat = Classifier_AB.predict(X_test)
# [fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))