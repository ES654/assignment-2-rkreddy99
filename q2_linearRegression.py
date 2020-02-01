import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression

from metrics import *

np.random.seed(42)

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
    #LR.plot()

    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
