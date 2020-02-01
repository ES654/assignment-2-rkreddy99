import pandas as pd
import numpy as np
X = pd.read_csv("iris.data")
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
Xtemp = X.copy().sample(frac=1,replace=True)
y = pd.Series([i for i in range(X.shape[0])])
Xtemp['label'] = y.copy()
X1 = Xtemp.copy().sample(frac=1,replace=True).reset_index(drop=True)
print(X1)
y1 = X1['label'].copy()
print(sum(y1))
X1 = X1.drop(['label'], axis=1)
print(X1)

# X = X.drop(["sepal length"], axis=1)
# X = X.drop(["petal length"], axis=1)


# for i in range(X["label"].size):
#     if X["label"][i]!="Iris-virginica":
#         X["label"][i] = "Iris-non-virginica"
# ind = [i for i in range(X.shape[0])]
# np.random.shuffle(ind)
# Xtemp = X.copy()

# for i in range(len(ind)):
#     X.loc[i] = Xtemp.loc[ind[i]]

# X_train = X.loc[:89].reset_index(drop=True)
# y_train = X_train["label"]
# X_train = X_train.drop(["label"], axis=1)

# X_test = X.loc[90:].reset_index(drop=True)
# y_test = X_test["label"]
# X_test = X_test.drop(["label"], axis=1)
