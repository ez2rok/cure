from sklearn.datasets import load_iris
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch
from icecream import ic

BATCH_SIZE = 2500
IMG_WIDTH = IMG_HEIGHT = 28


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


def download_fashion_mnist(dir, train=True):

    os.system('mkdir -p {}'.format(dir))
    data = datasets.FashionMNIST(
        dir,
        download=True,
        train=train,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    os.system('rm -rf {}/FashionMNIST/raw/*.gz'.format(dir))

    return data


def get_mean_std(data, batch_size):
    """
    Dynamically compute the mean and standard deviation of the data
    by only loading a small batch of data into memory.

    Parameters
    ----------
    data : (n_samples, IMG_WIDTH, IMG_HEIGHT) tensor
        The images.
    batch_size : int
        The size of the batch.
    """

    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    num_pixels = len(data) * IMG_WIDTH * IMG_HEIGHT

    mean = np.sum([batch[0].sum() for batch in loader]) / num_pixels
    std = np.sum([(batch[0].sum() - mean).pow(2).sum()
                  for batch in loader]) / num_pixels

    return mean, std


def normalize(data, mean, dir, train=True, std=1):
    """
    Wang's Paper says "The inputs for CURE and other methods are raw images and their pixel-wise centered versions". 
    According to https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/,
    Pixel Centering means we "scale pixel values to have a zero mean".
    We can do (x - mean) / std where we just set std = 1.
    """

    data_normal = datasets.FashionMNIST(
        dir,
        download=True,
        train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )
    loader = DataLoader(data_normal, batch_size=len(data), shuffle=False)
    data = next(iter(loader))
    return data


def get_classes(idxs, train_set, n_class1, n_class2):

    # initial values
    X1 = np.empty((n_class1, IMG_WIDTH, IMG_HEIGHT))
    X2 = np.empty((n_class2, IMG_WIDTH, IMG_HEIGHT))
    y1 = np.empty(n_class1)
    y2 = np.empty(n_class2)

    labels_map = {
        "T-Shirt": 0,
        "Trouser": 1,
        "Pullover": 2,
        "Dress": 3,
        "Coat": 4,
        "Sandal": 5,
        "Shirt": 6,
        "Sneaker": 7,
        "Bag": 8,
        "Ankle Boot": 9
    }

    imgs = train_set[0]
    labels = train_set[1]
    idxs = [labels_map[idx] for idx in idxs] if type(idxs[0]) == str else idxs

    # loop through all datapoints and select the ones that belong to the
    # desired classes
    i1, i2 = 0, 0
    for i, (img, label) in enumerate(zip(imgs, labels)):
        if i % 50 == 0:
            print("{}/{}".format(i, len(imgs)))

        if label == idxs[0] and i1 < n_class1:
            X1[i1] = img
            y1[i1] = label
            i1 += 1
        elif label == idxs[1] and i2 < n_class2:
            X2[i2] = img
            y2[i2] = label
            i2 += 1

        if i1 >= n_class1 and i2 >= n_class2:
            break

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    return X, y


def save_data(X, y, dir):

    data_path = '{}/FashionMNIST/processed/data.npy'.format(dir)
    label_path = '{}/FashionMNIST/processed/labels.npy'.format(dir)
    os.system('mkdir -p {}/FashionMNIST/processed'.format(dir))
    np.save(data_path, X)
    np.save(label_path, y)


def fashion_mnist_data(dir, idxs, batch_size=BATCH_SIZE, n_class1=6000, n_class2=2000):

    # download and save the raw data
    print('Downloading the Fashion MNIST data...')
    train_set = download_fashion_mnist(dir)
    print('Finished downloading the Fashion MNIST data.')

    # normalize the data
    print('Normalizing the Fashion MNIST data...')
    mean, std = get_mean_std(train_set, batch_size)
    train_set = normalize(train_set, mean, dir, std=std)
    print('Finished normalizing the Fashion MNIST data.')

    # extract and save the desired classes
    print('Extracting the desired classes...')
    X, y = get_classes(idxs, train_set, n_class1, n_class2)
    save_data(X, y, dir)
    print('Finished extracting the desired classes.')

    return X, y