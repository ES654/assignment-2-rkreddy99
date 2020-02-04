import numpy as np
import matplotlib.pyplot as plt
class LinearRegression():
    def __init__(self, fit_intercept=True, method='normal'):
        '''

        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        :param method:
        '''
        self.fit_intercept = fit_intercept
        self.method = method
        self.theta = 0

    def fit(self, X, y):
        '''
        Function to train and construct the LinearRegression
        :param X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        :param y: pd.Series with rows corresponding to output variable (shape of Y is N)
        :return:
        '''
        self.y1 = y
        X = X.to_numpy()
        if self.fit_intercept:
            a = np.ones((X.shape[0],1))
            X = np.concatenate((a,X), axis=1)
        inv = np.linalg.pinv(X.T @ X)
        self.theta = inv @ X.T @ y
        return self.theta


    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point
        :param X: pd.DataFrame with rows as samples and columns as features
        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X = X.to_numpy()
        if self.fit_intercept:
            a = np.ones((X.shape[0],1))
            X = np.concatenate((a,X), axis=1)
        self.prediction = X @ self.theta
        return self.prediction 
        

    def plot_residuals(self,fold):
        """
        Function to plot the residuals for LinearRegression on the train set and the fit. This method can only be called when `fit` has been earlier invoked.

        This should plot a figure with 1 row and 3 columns
        Column 1 is a scatter plot of ground truth(y) and estimate(\hat{y})
        Column 2 is a histogram/KDE plot of the residuals and the title is the mean and the variance
        Column 3 plots a bar plot on a log scale showing the coefficients of the different features and the intercept term (\theta_i)

        """
        y_hat = np.array(self.prediction)
        y = np.array(self.y1)
        fig, (ax1,ax2, ax3) = plt.subplots(1,3,figsize=(13,3))
        y,y_hat = np.array(y),np.array(y_hat)
        ax1.scatter(y,y_hat)
        ax1.set_xlabel('y')
        ax1.set_ylabel('y_hat')
        ax1.set_title('y_hat vs y',fontsize=10)
        exp_data = (y_hat-y)
        ax2.hist(exp_data, bins=30, align='left', color='r', edgecolor='k',linewidth=1)
        ax2.set_xlabel("Error")
        ax2.set_ylabel("No. of points with that error")
        ax2.set_title('mean: '+str(round(np.mean(exp_data),4)) +' '+'variance: '+str(round(np.var(exp_data),4)) +')',fontsize=10)
        theta = self.theta
        ax3.bar([str(i) for i in range(len(theta))],[abs(j) for j in theta])
        ax3.set_yscale('log')
        ax3.set_xlabel("Thetas")
        ax3.set_ylabel("Coefficients of Theta")
        ax3.set_title("Thetas (log scale)",fontsize=10)
        plt.tight_layout()
        if fold!=-1:
            plt.suptitle('Res. Plots of LR on RealEst. data for fold: '+str(fold)+","+str(self.fit_intercept),fontsize=12)
        else:
            plt.suptitle('Residual Plots of Linear Regression on random data'+","+str(self.fit_intercept),fontsize=12)
        plt.subplots_adjust(right = 1)
        return fig
