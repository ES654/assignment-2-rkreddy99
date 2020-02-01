"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index

np.random.seed(42)

class DecisionTree():
    def __init__(self, criterion):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.criterion = criterion
        self.tree = {}

    def fit(self, X, y, W):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        h = list(X.columns)
        X0, X1 = X[h[0]].sort_values(), X[h[1]].sort_values()
        s=[]
        l = len(X0)
        N = X
        N["weight"] = W
        N["output"] = list(y)

        for i in range(l-1):
            div = (X0[i] + X0[i+1])/2
            s.append(information_gain(N, h[0], div))

        for i in range(l-1):
            div = (X1[i] + X1[i+1])/2
            s.append(information_gain(N, h[1], div))

        div_ind = s.index(max(s))%(l-1) 
        div_attr = h[s.index(max(s))//(l-1)]

        X_sort = X[div_attr].sort_values()
        div_val = (X_sort[div_ind] + X_sort[div_ind + 1])/2
        
        Xd11 = N.loc[N[div_attr] <= div_val]
        Xd22 = N.loc[N[div_attr] > div_val]

        output1 = list(Xd11["output"])
        output2 = list(Xd22["output"])

        op1 = max(set(output1), key=output1.count)
        op2 = max(set(output2), key=output2.count)

        self.tree[div_attr] = [div_val, op1, op2]
        return self.tree
    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat = []
        tree = self.tree
        a = tree.keys()
        for i in a:
            for j in range(len(X[i])):
                if X[i][j] <= tree[i][0]:
                    y_hat.append(tree[i][1])
                else:
                    y_hat.append(tree[i][2])
        return y_hat

    def plot(self,X, y, W):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        tree = self.fit(X, y, W)
        a = tree.keys()
        for i in a:
            print("?(" + str(i) + " <= " +str(tree[i][0]) +")")
            print("\t" +"Y: Class "+str(tree[i][1]))
            print("\t" +"N: Class "+str(tree[i][2]))

        pass
