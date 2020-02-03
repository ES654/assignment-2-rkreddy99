"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)

########### RandomForestClassifier ###################
print("---------------------------------------")
print("RandomForestClassifier on random data")
print("---------------------------------------")

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")
n_estimators = 10
for criteria in ['entropy', 'gini']:
    Classifier_RF = RandomForestClassifier(n_estimators=n_estimators, criterion = criteria, max_depth=None)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    Classifier_RF.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

########### RandomForestRegressor ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
print("---------------------------------------")
print("RandomForestRegressor on random data")
print("---------------------------------------")

Regressor_RF = RandomForestRegressor(10, criterion = criteria)
Regressor_RF.fit(X, y)
y_hat = Regressor_RF.predict(X)
Regressor_RF.plot()
print('Criteria : MAE' )
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
