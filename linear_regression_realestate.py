import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

X = pd.read_excel("Real_estate_data.xlsx")
X.columns = ['No','trans_date', 'house_age',
       'dist_2_MRT',
       'num_stores', 'lat', 'longi',
       'price_per_unit_area']
y = X['price_per_unit_area']
X = X.drop(['price_per_unit_area'], axis=1)
X.reset_index(drop=True)
y.reset_index(drop=True)
print('--------------------------------------------')
print('Linear regression on real estate data')
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
