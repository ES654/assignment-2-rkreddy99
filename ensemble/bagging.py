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
        self.a=[]
        for i in range(self.n_estimators):
            clone = copy.deepcopy(self.base_estimator)
            X1 = pd.DataFrame({}, columns = [i for i in X])
            y1 = pd.Series([])
            ind1 = pd.Series([i for i in range(X.shape[0])]).sample(frac=1, replace=True, random_state=42).reset_index(drop=True)
            # print(ind1)
            for j in range(X.shape[0]):
                # ind = np.random.randint(0,X.shape[0]-1)
                X1.loc[j], y1.loc[j] = X.iloc[ind1[j]], y.iloc[ind1[j]]
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
        '''h = .02  # step size in the mesh

        names = ["Decision Tree"]

        classifiers = [DecisionTreeClassifier(max_depth=1)]

        df, ot = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                random_state=1, n_clusters_per_class=1)
        # rng = np.random.RandomState(2)
        # df += 2 * rng.uniform(size=df.shape)
        linearly_separable = (df, ot)

        datasets = [make_moons(noise=0.3, random_state=0),
                    make_circles(noise=0.2, factor=0.5, random_state=1),
                    linearly_separable
                    ]

        figure = plt.figure(figsize=(27, 9))
        i = 1
        # iterate over datasets
        for ds_cnt, ds in enumerate(datasets):
            # preprocess dataset, split into training and test part
            df, ot = ds
            df = StandardScaler().fit_transform(df)
            df_train, df_test, y_train, y_test = \
                train_test_split(df, ot, test_size=.4, random_state=42)

            x_min, x_max = df[:, 0].min() - .5, df[:, 0].max() + .5
            y_min, y_max = df[:, 1].min() - .5, df[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))

            # just plot the dataset first
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(['#FF0000', '#0000FF'])
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            if ds_cnt == 0:
                ax.set_title("Input data")
            # Plot the training points
            ax.scatter(df_train[:, 0], df_train[:, 1], c=y_train, cmap=cm_bright,
                    edgecolors='k')
            # Plot the testing points
            ax.scatter(df_test[:, 0], df_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                    edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            i += 1

            # iterate over classifiers
            for name, clf in zip(names, classifiers):
                ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
                clf.fit(df_train, y_train)
                score = clf.score(df_test, y_test)

                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, x_max]x[y_min, y_max].
                if hasattr(clf, "decision_function"):
                    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                else:
                    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

                # Plot the training points
                ax.scatter(df_train[:, 0], df_train[:, 1], c=y_train, cmap=cm_bright,
                        edgecolors='k')
                # Plot the testing points
                ax.scatter(df_test[:, 0], df_test[:, 1], c=y_test, cmap=cm_bright,
                        edgecolors='k', alpha=0.6)

                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())
                if ds_cnt == 0:
                    ax.set_title(name)
                ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                        size=15, horizontalalignment='right')
                i += 1

        plt.tight_layout()
        plt.show()'''
        pass
