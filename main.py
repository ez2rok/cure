from numpy.core.records import record
from icecream import ic
import numpy as np

# import local files
from make_dataset import iris_data
from cure import CURE
from objective import get_discriminant, objective

classes = ['setosa', 'virginica']
X, y = iris_data(classes)

a, b = 1.1, 2
cure = CURE(a, b, random_state=42)

weights, weight_history = cure.fit(X, record_history=True)
ic(weights, weight_history)