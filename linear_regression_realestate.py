import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

X = pd.read_excel("Real_estate_data.xlsx")
X = X.drop(['No'], axis=1)
X.columns = ['trans_date', 'house_age',
       'dist_2_MRT',
       'num_stores', 'lat', 'longi',
       'price_per_unit_area']
y = X['price_per_unit_area'].reset_index(drop=True)
X = X.drop(['price_per_unit_area'], axis=1).reset_index(drop=True)
X.reset_index(drop=True)
y.reset_index(drop=True)

print('===========================================================================')
print('Linear regression on real estate data')
print("===========================================================================")
l = X.shape[0]
l1 = X.shape[0]//5
y_hat = []
for fit_intercept in [True, False]:
    print('===============================================')
    print('fit_intercept : '+str(fit_intercept))
    print('===============================================')
    for i in range(5):
        X1 = X
        y1 = y
        Xtest = X1.loc[l1*i:l1*(i+1)-1].reset_index(drop=True)
        ytest = y1.loc[l1*i:l1*(i+1)-1].reset_index(drop=True)
        Xtrain = X1.drop([i for i in range(l1*i,l1*(i+1))]).reset_index(drop=True)
        ytrain = y1.drop([i for i in range(l1*i,l1*(i+1))]).reset_index(drop=True)
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit(Xtrain, ytrain)
        y_hat1 = LR.predict(Xtrain)
        y_hat = LR.predict(Xtest)
    #LR.plot()
        print('-------------------------------')
        print('errors for fold '+str(i+1))
        print('-------------------------------')
        y_hat = pd.Series(y_hat)
        print('MAE on train data : ', mae(y_hat1, ytrain))
        print('MAE on test data : ', mae(y_hat, ytest))
