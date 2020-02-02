from sklearn.tree import DecisionTreeClassifier
from linearRegression.linearRegression import LinearRegression
import pandas as pd
import numpy as np
import copy
import random
np.random.seed(42)
random.seed(42)
class RandomForestClassifier():
    def __init__(self, n_estimators, criterion, max_depth):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.tree = DecisionTreeClassifier(criterion=self.criterion)
        
    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.a=[]
        self.node = []
        for i in range(self.n_estimators):
            clone = copy.deepcopy(self.tree)
            Xtemp = X.copy()
            Xtemp['kljh'] = y.copy()
            X1 = Xtemp.copy().sample(frac=1,replace=True,random_state=42).reset_index(drop=True)
            y1 = X1['kljh'].copy()
            X1 = X1.drop(['kljh'], axis=1)
            col = list(X.columns)
            num = int(len(col)**0.5)
            attr = random.sample(col,num)
            self.node.append(attr)
            X2 = X1[attr].copy()
            clone.fit(X2,y1)
            self.a.append(clone)
    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pred = []
        y_hat = []
        tree = self.a
        assert(len(self.node)==len(self.a))
        for i in range(len(self.node)):
            attr = self.node[i]
            X2 = X[attr].copy()
            pred.append(list(tree[i].predict(X2)))
        for i in range(len(pred[0])):
            d = {}
            for j in range(len(pred)):
                if pred[j][i] in d:
                    d[pred[j][i]]+=1
                else:
                    d[pred[j][i]] = 1
            y_hat.append(max(d,key=d.get))
        y_hat = pd.Series(y_hat)
        return y_hat

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        pass



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators =n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.fit_intercept = True
        self.tree = LinearRegression(fit_intercept=self.fit_intercept, method='normal')

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.a=[]
        self.node = []
        for i in range(self.n_estimators):
            clone = copy.deepcopy(self.tree)
            X1 = pd.DataFrame({}, columns = [i for i in X])
            y1 = pd.Series([])
            for j in range(X.shape[0]):
                ind = random.randint(0,X.shape[0]-1)
                X1.loc[j], y1.loc[j] = X.iloc[ind], y.iloc[ind]
            col = list(X.columns)
            num = int(len(col)**0.5)
            attr = random.sample(col,num)
            self.node.append(attr)
            X2 = X[attr].copy()
            self.a.append(clone.fit(X2,y1))

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pred = []
        y_hat = []
        tree = self.a
        for i in range(len(self.node)):
            attr = self.node[i]
            X2 = X[attr].copy()
            X2 = X2.to_numpy()
            if self.fit_intercept:
                a = np.ones((X2.shape[0],1))
                X2 = np.concatenate((a,X2), axis=1)
            pred.append(list(X2 @ tree[i]))
        pred = np.array(pred)
        y_hat = sum(pred)/len(self.node)
        y_hat = pd.Series(y_hat)
        return y_hat

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        pass
