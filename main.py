from icecream import ic
from sklearn.model_selection import train_test_split as tts
import os
import numpy as np
from time import perf_counter


# import local files
from make_dataset import iris_data, fashion_mnist_data, add_intercept, synthetic_elliptical_data
from cure import CURE
import evaluate as eval

# initial values
seed = 420

# get data
classes = [2, 1]
X, y = iris_data(classes)
X_train, X_test, y_train, y_test = tts(X, y, random_state=seed)

# start = perf_counter()
# X, y = fashion_mnist_data()
# X = add_intercept(X)
# end = perf_counter()
# print('Time to load data: {}'.format(end - start))
# X_train, X_test, y_train, y_test = tts(X, y, random_state=seed)

# perform cure and make predictions
start = perf_counter()
cure = CURE(random_state=seed)
cure.fit(X_train, n_starts=1)
y_pred = cure.predict(X_test)
end = perf_counter()
print('Time to run cure: {}'.format(end - start))

# evaluate the predictions
ic(y_test.shape, y_pred.shape)
adj_rand = eval.adjusted_rand(y_test, y_pred) # adj_rand ∈ [-1, 1]
acc = eval.misclassification_rate(y_test, y_pred) # acc ∈ [0, 1]
ic(adj_rand, acc)


# n = 100
# d = 2
# seed = 420
# X = synthetic_elliptical_data(n, d, seed)