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

X = pd.read_csv("iris.data")
# 
a = []
for i in range(5):
  if i==4:
    a.append(X.columns[i])
  else:
    a.append(float(X.columns[i]))
print(a)
col = ["sepal length", "sepal width", "petal length", "petal width", "label"]
X.columns = col
# print(X)
# d={}
# for i in range(5):
#   d[col[i]] = a[i]
# X = X.append(d, ignore_index=True)
X.loc[-1] = a
X.index = X.index+1
X = X.sort_index()

X = X.drop(["sepal length"], axis=1)
X = X.drop(["petal length"], axis=1)
print(X)
y = X['label'].copy()
for i in range(y.size):
    if y[i]!="Iris-virginica":
        y[i] = "Iris-non-virginica"

X = X.drop(["label"], axis=1)
X['label'] = y.copy()

ind = [i for i in range(X.shape[0])]

np.random.shuffle(ind)
Xtemp = X.copy()
for i in range(len(ind)):
    X.loc[i] = Xtemp.loc[ind[i]]
X = X.reset_index(drop=True)
y = X["label"]

X_train = X.loc[:89].reset_index(drop=True)
y_train = X_train["label"]
X_train = X_train.drop(["label"], axis=1)

X_test = X[90:].reset_index(drop=True)
y_test = X_test["label"]
X_test = X_test.drop(["label"], axis=1)
