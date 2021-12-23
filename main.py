from numpy.core.records import record
from icecream import ic
import numpy as np

# import local files
from make_dataset import iris_data
from model import minimize_obj
from objective import get_discriminant, objective

classes = ['setosa', 'virginica']
X, y = iris_data(classes)

data = X
a, b = 1.1, 2
weights, weight_history = minimize_obj(data, a, b, record_history=True, random_state=42)
score = objective(weights, data, a, b)
ic(score, weights, weight_history.shape)