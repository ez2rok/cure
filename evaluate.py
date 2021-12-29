from sklearn.metrics import adjusted_rand_score
import numpy as np
from icecream import ic

def adjusted_rand(y_test, y_pred):
    adj_rand = adjusted_rand_score(y_test, y_pred)
    return adj_rand

def misclassification_rate(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)