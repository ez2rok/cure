from sklearn.metrics import adjusted_rand_score
import numpy as np
from icecream import ic

def adjusted_rand(y_test, y_pred):
    """
    Compute the adjusted rand index.

    Parameters
    ----------
    y_test : (n_samples,) array
        The true labels.
    y_pred : (n_samples,) array
        The predicted labels.

    Returns
    -------
    adj_rand : float
        The adjusted rand index. Values will be in between -1 and 1. The higher the better.
    """    
    adj_rand = adjusted_rand_score(y_test, y_pred)
    return adj_rand

def misclassification_rate(y_test, y_pred):
    """
    Compute the misclassification rate. 
    
    Because this is a clustering problem, it is possible that one class in y_test is labeled 
    as a different class in y_pred. To fix this, we compute the misclassification rate for all 
    permutations of y_test and y_pred's labels and select the label arrangement with the lowest
    misclassification rate. This is a binary classification problem, so we only need to 
    switch the labels once.

    Parameters
    ----------
    y_test : (n_samples,) array
        The true labels.
    y_pred : (n_samples,) array
        The predicted labels.

    Returns
    -------
    float
        The misclassification rate.
    """    
    
    score1 = 1 - np.sum(y_test != y_pred) / len(y_test)
    score2 = 1 - np.sum(y_test == y_pred) / len(y_test)
    return min(score1, score2)