import math
from tree.base import DecisionTree
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.trees = []
        self.alph=[]
        self.weight = []

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X.copy()
        self.y = y.copy()
        l = X.shape[0]
        weights = [1/l]*l
        
        for i in range(self.n_estimators):
            self.y1 = y.copy()
            clone = copy.deepcopy(self.base_estimator)
            tr = clone.fit(X,y,weights)
            y_hat = pd.Series(clone.predict(X))
            w=sum(weights)
            self.df = list(X.columns)
            self.x1 = X[self.df[0]].copy()
            self.x2 = X[self.df[1]].copy()
            self.weight.append(list(np.array(weights)*4000))
            assert(y_hat.size == y.size)
            indw = [] #wrongly predicted indices
            indt = [] #correctly predicted indices

            for j in range(y_hat.size):
                if y[j]!=y_hat[j]:
                    indw.append(j)
                else:
                    indt.append(j)
            err = 0
            for i in indw:
                err+=(weights[i]/w)
            # print(err)
            alpha = 0.5*math.log((1-err)/err)
            # print(weights)
            for i in indt:
                weights[i] = weights[i]*math.exp(-1*alpha)
            for i in indw:
                weights[i] = weights[i]*math.exp(alpha)
            weig = sum(weights)
            we = np.array(weights)/weig
            weights = list(we)
            self.trees.append(alpha)
            self.alph.append(alpha)
            self.trees.append(tr)
       
    
    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pred = []
        y_hat = []
        for i in range(0,self.n_estimators*2,2):
            yhat = []
            tree = self.trees[i+1]
            a = tree.keys()
            for k in a:
                for j in range(len(X[k])):
                    if X[k][j] <= tree[k][0]:
                        yhat.append(tree[k][1])
                    else:
                        yhat.append(tree[k][2])
            pred.append(yhat)
        for i in range(len(pred[0])):
            d = {}
            for j in range(len(pred)):
                if pred[j][i] in d:
                    d[pred[j][i]]+=self.alph[j]
                else:
                    d[pred[j][i]] = self.alph[j]
            y_hat.append(max(d,key=d.get))
        y_hat = pd.Series(y_hat)
        return y_hat

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        cla = list(self.y1.unique())
        tree = self.trees
        x1 = list(self.x1)
        x2 = list(self.x2)
        y1 = list(self.y1)
        fig,axes = plt.subplots(1,self.n_estimators)
        for ii in range(self.n_estimators):
            a = []
            b = []
            c = []
            d = []
            w1 = []
            w2 = []
            for i in range(len(x1)):
                if y1[i]==cla[0]:
                    a.append(x1[i])
                    b.append(x2[i])
                    w1.append(self.weight[ii][i])
                else:
                    c.append(x1[i])
                    d.append(x2[i])
                    w2.append(self.weight[ii][i])
            axes[ii].scatter(np.array(a), np.array(b), s=w1, c='#0001fb', edgecolors='k')
            axes[ii].scatter(np.array(c), np.array(d), s=w2, c='#eb170a', edgecolors='k')
            axes[ii].set_xlim(min(x1)-0.25,max(x1)+0.25)
            axes[ii].set_ylim(min(x2)-0.25,max(x2)+0.25)
            for i in tree[2*ii+1]:
                if i == self.df[0]:
                    axes[ii].axvspan(min(x1)-1,tree[2*ii+1][i][0],facecolor='#93b7d7',alpha=0.5)
                    axes[ii].axvspan(tree[2*ii+1][i][0],max(x1)+1,facecolor='#db9397',alpha=0.5)
                else:
                    axes[ii].axhspan(min(x2)-1,tree[2*ii+1][i][0],facecolor='#93b7d7',alpha=0.5)
                    axes[ii].axhspan(tree[2*ii+1][i][0],max(x2)+1,facecolor='#db9397',alpha=0.5)
                
        plt.show()
        plt.close()

        plot_colors = 'rb'
        plot_step = 0.02
        X = self.X.copy()
        cols=X.columns
        y = self.y.copy()
        cols = list(X.columns)
        x_min, x_max = X.iloc[:, 0].min() - 0.25, X.iloc[:, 0].max() + 0.25
        y_min, y_max = X.iloc[:, 1].min() - 0.25, X.iloc[:, 1].max() + 0.25
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                            np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        # clf = classifiers[i]
        X_ = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(pd.DataFrame({cols[i]: pd.Series(X_[:,i]) for i in range(len(X_[0]))}))
        Z = np.array(Z).reshape(xx.shape)
        try:
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PuOr)
        except:
            for i in range(len(Z)):
                Z[i] = [int(j=='Iris-virginica') for j in Z[i]]
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.PuOr)

        # plt.xlabel()
        # plt.ylabel(iris.feature_names[pair[1]])
        plt.title("Fig 2")

        # Plot the training points
        for cls, color in zip(np.unique(y), plot_colors):
            # print(color)
            # break
            idx = np.where(y == cls)[0]
            plt.scatter(X.iloc[idx, 0], X.iloc[idx, 1], c=color, cmap=plt.cm.PuOr, edgecolor='black', s=50)
        plt.show()

        return
