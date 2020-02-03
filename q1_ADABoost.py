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
print("Decision stump on random data")
print("-----------------------------------------------------------")
N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria)
re = X.shape[0]
img_weights = [1/re]*re
tree.fit(X,y,img_weights)
yhat = pd.Series(tree.predict(X))
print('Criteria :', criteria)
print('Accuracy: ', accuracy(yhat, y))
for cls in y.unique():
    print("***Class :"+str(cls)+"***")
    print('Precision: ', precision(yhat, y, cls))
    print('Recall: ', recall(yhat, y, cls))

print("-----------------------------------------------------------")
print("Adaboost on random data")
print("-----------------------------------------------------------")

Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
#[fig1, fig2] = 
Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print("***Class :"+str(cls)+"***")
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features
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

X.loc[-1] = a
X.index = X.index+1
X = X.sort_index()

X = X.drop(["sepal length"], axis=1)
X = X.drop(["petal length"], axis=1)

y = X['label'].copy()
for i in range(y.size):
    if y[i]!="Iris-virginica":
        y[i] = "Iris-non-virginica"

X = X.drop(["label"], axis=1)
X['label'] = y.copy()

X = X.sample(frac=1,random_state=42)
X = X.reset_index(drop=True)
y = X["label"]

X_train = X.loc[:89].reset_index(drop=True)
y_train = X_train["label"]
X_train = X_train.drop(["label"], axis=1)

X_test = X[90:].reset_index(drop=True)
y_test = X_test["label"]
X_test = X_test.drop(["label"], axis=1)
print("-----------------------------------------------------------")
print("Decision stump on IRIS data")
print("-----------------------------------------------------------")

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria)
re = X_train.shape[0]
img_weights = [1/re]*re
tree.fit(X_train,y_train,img_weights)
yhat = pd.Series(tree.predict(X_test))
print('Criteria :', criteria)
print('Accuracy: ', accuracy(yhat, y_test))
for cls in y.unique():
    print("***Class :"+str(cls)+"***")
    print('Precision: ', precision(yhat, y_test, cls))
    print('Recall: ', recall(yhat, y_test, cls))
print("-----------------------------------------------------------")
print("Adaboost on iris data")
print("-----------------------------------------------------------")


n_estimators = 3

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X_train, y_train)
y_hat = Classifier_AB.predict(X_test)
# [fig1, fig2] = 
Classifier_AB.plot()
print('***Criteria :'+str(criteria)+"***")
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print("***Class :"+str(cls)+"***")
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))