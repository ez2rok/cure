from sklearn.metrics import adjusted_rand_score, accuracy_score
import numpy as np
from icecream import ic

def adjusted_rand(y_test, y_pred):
    adj_rand = adjusted_rand_score(y_test, y_pred)
    return adj_rand

def misclassification_rate(y_test, y_pred):
    score1 = 1 - np.sum(y_test != y_pred) / len(y_test)
    score2 = 1 - np.sum(y_test == y_pred) / len(y_test)
    return min(score1, score2)