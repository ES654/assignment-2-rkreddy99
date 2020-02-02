import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.xlim(0,9)
plt.ylim(0,9)

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
print(x3)

w = []
for i in range(len(x3)):
	w.append(np.random.randint(10,30))
w = np.array(w)
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
for i in range(len(w)):
	if x3[i]==1:
		plt.axvspan(i-.5,i+0.5, facecolor='b', alpha=0.5)
		plt.axhspan(i-.5,i+0.5, facecolor='b', alpha=0.5)
	else:
		plt.axvspan(i-.5,i+0.5, facecolor='y', alpha=0.5)
		plt.axhspan(i-.5,i+0.5, facecolor='y', alpha=0.5)
# plt.axvspan(5,10, facecolor='y', alpha=0.5)
# plt.axvspan(5,10, facecolor='y', alpha=0.5)
plt.scatter(x1,x2,s=w)
plt.show()

# import pandas as pd
# import numpy as np
# import random
# X = pd.read_csv("iris.data")
# a = []
# for i in range(5):
#   if i==4:
#     a.append(X.columns[i])
#   else:
#     a.append(float(X.columns[i]))
# # col = ["sepal length", "sepal width", "petal length", "petal width", "label"]
# # X.columns = col
# # d={}
# # for i in range(5):
# #   d[col[i]] = a[i]
# # X = X.append(d, ignore_index=True)
# # Xtemp = X.copy().sample(frac=1,replace=True)
# y = pd.Series([i for i in range(X.shape[0])])
# # Xtemp['label'] = y.copy()
# # X1 = Xtemp.copy().sample(frac=1,replace=True).reset_index(drop=True)
# # print(X1)
# # y1 = X1['label'].copy()
# # print(sum(y1))
# # X1 = X1.drop(['label'], axis=1)
# # print(X1)

# Xtemp = X.copy()
# Xtemp['kljh'] = y.copy()
# print(Xtemp)
# X1 = Xtemp.copy().sample(frac=1,replace=True,random_state=42)
# print(X1)
# print(X1.loc[102])
# y1 = X1['kljh'].copy()
# X1 = X1.drop(['kljh'], axis=1)
# col = list(X.columns)
# num = int(len(col)**0.5)
# attr = random.sample(col,num)
# X2 = X1[attr].copy()
# # X = X.drop(["sepal length"], axis=1)
# # X = X.drop(["petal length"], axis=1)


# # for i in range(X["label"].size):
# #     if X["label"][i]!="Iris-virginica":
# #         X["label"][i] = "Iris-non-virginica"
# # ind = [i for i in range(X.shape[0])]
# # np.random.shuffle(ind)
# # Xtemp = X.copy()

# # for i in range(len(ind)):
# #     X.loc[i] = Xtemp.loc[ind[i]]

# # X_train = X.loc[:89].reset_index(drop=True)
# # y_train = X_train["label"]
# # X_train = X_train.drop(["label"], axis=1)

# # X_test = X.loc[90:].reset_index(drop=True)
# # y_test = X_test["label"]
# # X_test = X_test.drop(["label"], axis=1)
