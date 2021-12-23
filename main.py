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
weights, weight_history = minimize_obj(data, record_weight_history=True, random_state=42)
score = objective(weights, data)

ic(weights, weight_history)