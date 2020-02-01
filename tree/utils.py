import math

def entropy(Y,W):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    d = {}
    for i in range(len(Y)):
        if Y[i] in d:
            d[Y[i]] += W[i]
        else:
            d[Y[i]] = W[i]
    
    entro = 0
    w = sum(W)
    for i in d:
        entro += (-1*d[i]/w)*math.log(d[i]/w,2)
    
    return entro

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    pass

def information_gain(X, attr, val):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    Xd1 = X.loc[X[attr] <= val].reset_index(drop=True)
    Xd2 = X.loc[X[attr] > val].reset_index(drop=True)
    Y = list(X["output"])
    Y1 = list(Xd1["output"])
    Y2 = list(Xd2["output"])
    W = list(X["weight"])
    W1 = list(Xd1["weight"])
    W2 = list(Xd2["weight"])
    ig = entropy(Y,W) - (sum(W1)/sum(W))*entropy(Y1,W1) - (sum(W2)/sum(W))*entropy(Y2,W2)
    
    return ig
