from sklearn.datasets import load_iris
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch
from icecream import ic
import matplotlib.pyplot as plt

BATCH_SIZE = 10000
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
    std = [(batch[0] - mean).pow(2).sum() for batch in loader]
    std = np.sqrt(np.sum(std) / num_pixels)

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
    X1 = np.empty((n_class1, IMG_WIDTH * IMG_HEIGHT))
    X2 = np.empty((n_class2, IMG_WIDTH * IMG_HEIGHT))
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
            X1[i1] = img.flatten()
            y1[i1] = label
            i1 += 1
        elif label == idxs[1] and i2 < n_class2:
            X2[i2] = img.flatten()
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


def get_fashion_mnist_data(dir, idxs, batch_size, n_class1, n_class2):

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


def fashion_mnist_data(idxs=['T-Shirt', 'Pullover'], dir='./data', redownload=False,
                       batch_size=BATCH_SIZE, n_class1=6000, n_class2=6000):
    """
    Takes roughly 30 seconds to download, normalize, and extract the data.
    """

    data_path = '{}/FashionMNIST/processed/data.npy'.format(dir)
    label_path = '{}/FashionMNIST/processed/labels.npy'.format(dir)
    max_n_class1 = max_n_class2 = 6000

    if os.path.exists(data_path) and os.path.exists(label_path) and not redownload:
        X = np.load(data_path)
        y = np.load(label_path)
    else:
        X, y = get_fashion_mnist_data(
            dir, idxs, batch_size, max_n_class1, max_n_class2)

    X = np.concatenate(
        (X[:n_class1], X[max_n_class1: max_n_class1 + n_class2]), axis=0)
    y = np.concatenate(
        (y[:n_class1], y[max_n_class1: max_n_class1 + n_class2]), axis=0)
    return X, y


def add_intercept(X):
    """
    Adds a column of ones to the data matrix X.

    Parameters
    ----------
    X : (n_samples, n_features) array
        The data.

    Returns
    -------
    (n_samples, n_features + 1) array
        The data matrix X with a column of ones prepended.
    """

    return np.c_[np.ones(X.shape[0]), X]


def synthetic_elliptical_data(n, d, seed, mu_0_val=0, mu_val=[0, 8], sigma_val=1):
    """
    Synthetically generate data from an elliptical distribution
    according to the formula: X_i = mu_0 + mu y_i + sigma * z_i
    """

    # deterministic values
    mu_0 = np.full((n, d), mu_0_val)
    mu = np.full(d, mu_val)
    sigma = np.full((n, d, d), sigma_val * np.eye(d))

    # random values
    rng = np.random.default_rng(seed)
    y = rng.choice(np.array([1, -1]), size=n)
    mean = np.full(d, 0)
    cov = np.eye(d) * [10, 0.5]
    z = rng.multivariate_normal(mean, cov, size=n)[:, :, np.newaxis]

    # compute X with vectorized operations
    X = mu_0 + np.outer(y, mu) + np.einsum('ijk,ikl->ijl',
                                           np.sqrt(sigma), z).squeeze()

    params = {
        'mu_0': mu_0_val,
        'mu': mu_val,
        'sigma': sigma_val,
        'z': {'mean': mean, 'cov': cov},
    }

    return X, y, params


def plot_elliptical_data(X, y, params, outdir='figures/elliptical_data.png'):
    """
    Plot the data from the elliptical distribution.
    """

    # plot the data
    fig, ax = plt.subplots(figsize=(10, 5))
    for y_i in np.unique(y):
        ax.scatter(X[y == y_i, 0], X[y == y_i, 1], label=y_i, s=10)

    # format the data
    limit = np.max(np.abs(X))
    title = 'Synthetic Elliptical Data'
    ax.set(xlabel='x', ylabel='y', title=title,
           xlim=(-limit, limit), ylim=(-limit, limit))
    plt.legend()

    # save the data
    fig.savefig(outdir)
    return fig
