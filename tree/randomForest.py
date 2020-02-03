from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
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
        self.df = X.copy()
        self.ot = y.copy()
        self.node = []
        self.X = []
        self.y = []
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
            print(X2.shape)
            self.X.append(X2.copy())
            self.y.append(y1.copy())
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
        classifiers = [i for i in self.a]
        alphas = ['estimator '+str(i) for i in range(1,len(self.a)+1)]
        
        # print("in plot")
        # print(np.unique(y, return_counts=True)[1])
        for i in range(len(self.a)):
            X = np.array(self.X[i])
            y = np.array(self.y[i])
            # n_classes = 3
            plot_colors = 'rb'
            plot_step = 0.02

            plt.subplot(1, len(alphas), i + 1)
            
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                np.arange(y_min, y_max, plot_step))
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            clf = classifiers[i]
            X_ = np.c_[xx.ravel(), yy.ravel()]
            Z = clf.predict(pd.DataFrame({i: pd.Series(X_[:,i]) for i in range(len(X_[0]))}))
            Z = np.array(Z).reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PuOr)

            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.title(alphas[i])

            for cls, color in zip(np.unique(y), plot_colors):
                idx = np.where(y == cls)[0]
                plt.scatter(X[idx, 0], X[idx, 1], c=color, s = 40, cmap=plt.cm.PuOr, edgecolor='black')
        plt.show()

        plt.suptitle("Decision surface on bagged data")
        plt.close()

        plot_colors = 'rb'
        plot_step = 0.02

        # plt.subplot(1, len(alphas), i + 1)
        X = np.array(self.df)
        y = np.array(self.ot)
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                            np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        # clf = classifiers[i]
        X_ = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(pd.DataFrame({i: pd.Series(X_[:,i]) for i in range(len(X_[0]))}))
        Z = np.array(Z).reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PuOr)

        plt.xlabel('x1')
        plt.ylabel("x2")
        plt.title("Combined decision surface")

        # Plot the training points
        for cls, color in zip(np.unique(y), plot_colors):
            # print(color)
            # break
            idx = np.where(y == cls)[0]
            plt.scatter(X[idx, 0], X[idx, 1], c=color,cmap=plt.cm.PuOr, edgecolor='black', s=40)
        plt.show()


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
        self.tree = DecisionTreeRegressor(criterion='mae')


    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.a=[]
        self.node = []
        self.X = []
        self.y = []
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
            self.X.append(X2)
            self.y.append(y1)
            clone.fit(X2,y1)
            self.a.append(clone)

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
        assert(len(self.node)==len(self.a))
        for i in range(len(self.node)):
            attr = self.node[i]
            X2 = X[attr].copy()
            pred.append(list(tree[i].predict(X2)))
        y_hat = sum(np.array(pred))/len(pred[0])
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
