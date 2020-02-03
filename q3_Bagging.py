"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)
from metrics import *

from ensemble.bagging import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
# Or use sklearn decision tree
from linearRegression.linearRegression import LinearRegression

########### BaggingClassifier ###################
print("-------------------------------------------")
print("Bagging on random data")
print("-------------------------------------------")
N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 5
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")
# print(X)
tree = DecisionTreeClassifier(criterion="entropy",max_depth=None)
Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_B.fit(X, y)
print(X.shape)
y_hat = Classifier_B.predict(X)
Classifier_B.plot()
print('Criteria : entropy')
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))


print("-------------------------------------------")
print("Bagging on data from lecture")
print("-------------------------------------------")

x1 =[]
for i in range(1,9):
    for j in range(1,9):
        x1.append(i)
x2 = []
for i in range(1,9):
    for j in range(1,9):
        x2.append(j)
x3 = [-1 for i in range(64)]
for i in range(64):
	if x1[i]<=5 and x2[i]<=5:
		x3[i]=1
x3[18] = -1
x3[39] = 1
df = {}
df['x1'] = x1
df['x2'] = x2
df['label'] = x3
X = pd.DataFrame(df, columns = ['x1','x2','label'])
y = X['label'].reset_index(drop=True)
X = X.drop(['label'], axis=1).reset_index(drop=True)

tree = DecisionTreeClassifier(criterion="entropy",max_depth=None)
Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
Classifier_B.plot()
print('Criteria : entropy')
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))