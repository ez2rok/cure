from numpy.core.records import record
from icecream import ic
import numpy as np
from sklearn.model_selection import train_test_split as tts

# import local files
from make_dataset import iris_data
from cure import CURE
from objective import get_discriminant, objective

seed = 42
classes = ['setosa', 'virginica']
X, y = iris_data(classes)
X_train, X_test, y_train, y_test = tts(X, y, random_state=seed)

cure = CURE(random_state=seed)
y_pred = cure.fit_predict(X_train, record_history=False)