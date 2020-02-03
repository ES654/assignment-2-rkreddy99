import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier

np.random.seed(42)
print("============================================================")
print("Predicting on iris data")
print("============================================================")

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
col = X["label"].unique()
d = {}
a = 1
for i in col:
    if i not in d:
        d[i] = a
        a+=1
          
print(d)          
for i in range(X["label"].size):
    if X["label"][i] in d:
        X["label"][i] = d[X["label"][i]]

print(X)
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
X_test = X_test.drop(["label"], axis=1)

n_estimators = 5
for criteria in ['entropy', 'gini']:
    
    Classifier_RF = RandomForestClassifier(n_estimators=n_estimators, criterion = criteria, max_depth=None)
    
    Classifier_RF.fit(X_train, y_train)
    y_hat = Classifier_RF.predict(X_test)
    # Classifier_RF.plot()
    print('***Criteria :'+str(criteria)+"***")
  
    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y.unique():
        print("***Class :"+str(cls)+"***")
        print('Precision: ', precision(y_hat, y_test, cls))
        print('Recall: ', recall(y_hat, y_test, cls))