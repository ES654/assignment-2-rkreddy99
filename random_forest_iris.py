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
col = ["sepal length", "sepal width", "petal length", "petal width", "label"]
X.columns = col
X = X.sample(frac=1,random_state=40)
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
    fig, fig2 = Classifier_RF.plot()
    print('***Criteria :'+str(criteria)+"***")
  
    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y.unique():
        print("***Class :"+str(cls)+"***")
        print('Precision: ', precision(y_hat, y_test, cls))
        print('Recall: ', recall(y_hat, y_test, cls))
#plt.show()