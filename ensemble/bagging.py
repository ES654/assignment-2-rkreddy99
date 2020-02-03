import pandas as pd
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
np.random.seed(42)
class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.df = X.copy()
        self.ot = y.copy()
        self.X = []
        self.y = []
        self.a=[]
        for i in range(self.n_estimators):
            clone = copy.deepcopy(self.base_estimator)
            X1 = pd.DataFrame({}, columns = [i for i in X])
            y1 = pd.Series([])
            ind1 = pd.Series([i for i in range(X.shape[0])]).sample(frac=1, replace=True).reset_index(drop=True)
            # print(ind1)
            for j in range(X.shape[0]):
                # ind = np.random.randint(0,X.shape[0]-1)
                X1.loc[j], y1.loc[j] = X.iloc[ind1[j]], y.iloc[ind1[j]]
            self.X.append(X1.copy())
            self.y.append(y1.copy())
            self.a.append(clone.fit(X1,y1))

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pred = []
        y_hat = []
        for tree in self.a:
            pred.append(list(tree.predict(X)))
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
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        # X = self.X
        # y = self.y
        classifiers = [tem for tem in self.a]
        alphas = ['iter'+str(i) for i in range(1,len(self.a)+1)]
        
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
        plt.title("Ensemble of different estimators")

        # Plot the training points
        for cls, color in zip(np.unique(y), plot_colors):
            # print(color)
            # break
            idx = np.where(y == cls)[0]
            plt.scatter(X[idx, 0], X[idx, 1], c=color,cmap=plt.cm.PuOr, edgecolor='black', s=40)
        plt.show()
        
