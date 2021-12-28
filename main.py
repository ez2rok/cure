from icecream import ic
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import adjusted_rand_score
import os
import numpy as np

# import local files
from make_dataset import iris_data, fashion_mnist_data
from cure import CURE

# initial values
seed = 42

# # get data
# classes = [2, 1]
# X, y = iris_data(classes)
# X_train, X_test, y_train, y_test = tts(X, y, random_state=seed)

# # perform cure and make predictions
# cure = CURE(random_state=seed)
# cure.fit(X_train)
# y_pred = cure.predict(X_test)

# # evaluate the predictions
# adj_rand = adjusted_rand_score(y_test, y_pred) # adj_rand ∈ [-1, 1]
# ic(adj_rand)

dir = './data'
idxs = ['T-Shirt', 'Pullover']
n_class2 = 1000
X, y = fashion_mnist_data(dir, idxs, n_class2)
ic(np.unique(y, return_counts=True))
