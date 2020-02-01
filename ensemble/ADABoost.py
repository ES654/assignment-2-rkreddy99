import math
from tree.base import DecisionTree
import pandas as pd
import copy

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
        

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        l = X.shape[0]
        weights = [1/l]*l
        w=sum(weights)
        for i in range(self.n_estimators):
            clone = copy.deepcopy(self.base_estimator)
            tr = clone.fit(X,y,weights)
            y_hat = pd.Series(clone.predict(X))

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
            
            alpha = 0.5*math.log((1-err)/err)
            
            for i in indt:
                weights[i] = weights[i]*math.exp(-1*alpha)
            for i in indw:
                weights[i] = weights[i]*math.exp(alpha)
            
            self.trees.append(alpha)
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
                    d[pred[j][i]]+=1
                else:
                    d[pred[j][i]] = 1
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
        
        pass
