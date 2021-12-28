from sklearn.datasets import load_iris
import numpy as np
from torchvision import datasets, transforms
import os


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
    data = load_iris(as_frame=True)
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
    y = y[idxs].to_numpy()

    return X, y


def binary_encoding(y):
    """
    Encode the target values to binary encoding of -1, 1.

    Parameters
    ----------
    y : array (n_samples,)
        Target values.

    Returns
    -------
    [type]
        [description]
    """

    assert len(np.unique(y)) == 2, 'y must have only two classes'
    targets = [-1, 1]
    mapping = {y_val: target for y_val, target in zip(np.unique(y), targets)}
    y = np.array([mapping[y_val] for y_val in y])

    return y


def fashion_mnist(dir):

    os.system('mkdir -p {}'.format(dir))

    # Define a transform to convert to images to tensor
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the data: train and test sets
    dir = './data/raw'
    # trainset = datasets.FashionMNIST(dir, download=True, train=True,
    #                                  transform=transform)
    testset = datasets.FashionMNIST(dir, download=True, train=False,
                                    transform=transform)

    return testset
