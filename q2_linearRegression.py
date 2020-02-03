import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time
from metrics import *

np.random.seed(42)
N = [1000*i for i in range(1,6)]
P = [100*i for i in range(1,6)]
tt1 = []
tf1 = []
tt = []
tf = []
for i in range(len(N)):
    ttt = []
    tff = []
    ttt1 = []
    tff1 = []
    for j in range(len(P)):
        X = pd.DataFrame(np.random.randn(N[i], P[j]))
        y = pd.Series(np.random.randn(N[i]))
        for fit_intercept in [True, False]:
            if fit_intercept:
                a = time.time()
                LR = LinearRegression(fit_intercept=fit_intercept)
                LR.fit(X,y)
                y_hat = LR.predict(X)        
                b = time.time()
                ttt.append(b-a)
                ttt1.append(((N[i]**2)*(P[j]+1) + (P[j]+1)**3)*10**-9)
            else:
                a = time.time()
                LR = LinearRegression(fit_intercept=fit_intercept)
                LR.fit(X,y)
                y_hat = LR.predict(X)        
                b = time.time()
                tff.append(b-a)
                tff1.append(((N[i]**2)*(P[j]) + (P[j])**3)*10**-9)
    tt.append(ttt)
    tf.append(tff)
    tt1.append(ttt1)
    tf1.append(tff1)    
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier
fig, ax1 = plt.subplots(2,2)
fig.suptitle("Comparision of time complexity", fontsize=10)

ax = ax1[0][0]
im = ax.imshow(tt)

ax.set_xticks(np.arange(len(P)))
ax.set_yticks(np.arange(len(N)))
ax.set_xticklabels(P)
ax.set_yticklabels(N)
ax.set_ylabel('No. of samples')
ax.set_xlabel('No. of features')
plt.setp(ax.get_xticklabels(), ha="right")
we = np.array(tt)

for i in range(len(N)):
    for j in range(len(P)):
        text = ax.text(j, i, truncate(we[i, j],3),
                       ha="center", va="center", color="w")

ax.set_title("Observed time(s), fit_intercept: True")

ax = ax1[0][1]
im = ax.imshow(tf)

ax.set_xticks(np.arange(len(P)))
ax.set_yticks(np.arange(len(N)))
ax.set_xticklabels(P)
ax.set_yticklabels(N)
ax.set_ylabel('No. of samples')
ax.set_xlabel('No. of features')

plt.setp(ax.get_xticklabels(), ha="right")
we = np.array(tf)

for i in range(len(N)):
    for j in range(len(P)):
        text = ax.text(j, i, truncate(we[i, j],3),
                       ha="center", va="center", color="w")

ax.set_title("Observed time(s), fit_intercept: False")

ax = ax1[1][0]
im = ax.imshow(tt1)


ax.set_xticks(np.arange(len(P)))
ax.set_yticks(np.arange(len(N)))

ax.set_xticklabels(P)
ax.set_yticklabels(N)


plt.setp(ax.get_xticklabels(), ha="right")
we = np.array(tt1)

for i in range(len(N)):
    for j in range(len(P)):
        text = ax.text(j, i, truncate(we[i, j],3),
                       ha="center", va="center", color="w")

ax.set_title("Estimated time(s), fit_intercept: True")

ax = ax1[1][1]
im = ax.imshow(tf1)


ax.set_xticks(np.arange(len(P)))
ax.set_yticks(np.arange(len(N)))

ax.set_xticklabels(P)
ax.set_yticklabels(N)
ax.set_ylabel('No. of samples')
ax.set_xlabel('No. of features')

plt.setp(ax.get_xticklabels(), ha="right")
we = np.array(tf1)

for i in range(len(N)):
    for j in range(len(P)):
        text = ax.text(j, i, truncate(we[i, j],3),
                       ha="center", va="center", color="w")

ax.set_title("Estimated time(s), fit_intercept: False")
fig.tight_layout()
plt.show()

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
print('--------------------------------------------')
print('Linear regression on random data')
print('--------------------------------------------')
for fit_intercept in [True, False]:
    print('-------------------------')
    print('fit_intercept : '+str(fit_intercept))
    print('-------------------------')
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit(X, y)
    y_hat = LR.predict(X)
    LR.plot_residuals(fold = 1)

    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
