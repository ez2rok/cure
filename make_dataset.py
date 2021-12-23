from sklearn import datasets
import numpy as np


def iris_data(classes):
    """
    Download classes from the iris dataset.

    ---- Parameters ----
    classes (list): list of classes to download. Classes can be 
                    a list of numbers or strings where numbers
                    is the integer value assigned to a specific 
                    class and strings is the name of the class.
    """

    # download iris data
    data = datasets.load_iris(as_frame=True)
    X = data.data
    y = data.target

    # get int class labels for all the classes
    if classes == 'all':
        classes = np.unique(y)

    # convert str class labels to int class labels
    elif type(classes[0]) == str:
        mapping = {label: i for i, label in enumerate(data.target_names)}
        classes = [mapping[class_] for class_ in classes]


    # select the data and labels that belong to the specified classes
    idxs = np.ravel([np.where(y == class_)[0] for class_ in classes])
    X = X.iloc[idxs].to_numpy()
    y = y[idxs]

    return X, y