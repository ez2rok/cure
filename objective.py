import numpy as np
import matplotlib.pyplot as plt


def get_embedding(weights, X):
    """
    Embed X, often a multidimensional matrix, into 1D space via the dot product.

    """
    return np.dot(weights, X.T)


def get_discriminant(X, a, b):
    """
    Discriminative function that separates the data X into two clusters.

    Because f() has a minimum at both x=±1, minimizing f() will map many of our
    datapoints to x=1 and many of them to x=-1, resulting in two different clusters.

    Because h() becomes huge for large x values, we construct f() which is just
    like h() except when x is too big we clip its growth with linear functions 
    and a cubic spline interpolation.

    Note: 1 < a < b.
    if |x| <= a, f() has two valleys around ±1.
    if a < |x| <= b, f() is a cubic spline connecting the valleys and the linear function.
    if |x| > b, f() is a linear function.

    ---- Parameters ----
    embedding (n_samples): array of 1D embeddings
    a (float): threshold between h() and the cubic spline
    b (float): threshold between the cubic spline and the linear function

    """

    # initial values
    X_abs = np.abs(X)
    X_result = np.empty(X.shape)

    # define h = (x^2 - 1)^2 / 4 and its derivatives
    def h(x): return np.square(np.square(x) - 1) / 4
    def h_prime(x): return x * (np.square(x) - 1)
    def h_prime_prime(x): return 3 * np.square(x) - 1

    # case 1: |x| <= a
    idxs = np.where(X_abs <= a)
    X_result[idxs] = h(X[idxs])

    # case 2: a < |x| <= b
    idxs = np.where((a < X_abs) & (X_abs <= b))
    X_result[idxs] = h(a) + h_prime(a) * (X_abs[idxs] - a) \
        + h_prime_prime(a) * np.power(X_abs[idxs] - a, 2) / 2 \
        + h_prime_prime(a) * np.power(X_abs[idxs] - a, 3) / (6 * (b - a))

    # case 3: |x| > b
    idxs = np.where(X_abs > b)
    total = h(a) + h_prime(a) * (b - a) \
        + h_prime_prime(a) * np.power(b - a, 2) / 2 \
        + h_prime_prime(a) * np.power(b - a, 3) / (6 * (b - a))
    X_result[idxs] = total \
        + (h_prime(a) + 0.5 * (b - a) * h_prime_prime(a)) \
        * (X_abs[idxs] - b)

    return X_result


def plot_discriminant(outdir):
    """
    Plot the discriminant function.
    """

    # initial values
    a, b = 1.1, 1.5
    lower, upper = -2, 2
    x_values = np.linspace(lower, upper, 100)
    def h(x): return np.square(np.square(x) - 1) / 4

    # compute the discriminant function f and h
    f_values = get_discriminant(x_values[:], a=a, b=b)
    h_values = h(x_values[:])

    # plot figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, f_values, '.', label='f(x)')
    ax.plot(x_values, h_values, label='h(x)')
    ax.set(xlabel='x', ylabel='f(x)',
           title='Discriminant functions: f(x) and h(x) for a={}, b={}'.format(a, b))
    ax.legend()

    # save figure
    fig.savefig(outdir + '/discriminant.png')


def get_penalty(weights, X):
    """
    Compute the penalty term.

    The penalty term encourages the data X to be evenly split between
    the two clusters at x=±1. This prevents the objective function 
    from achieving the useless trivial solution of clustering all of
    the data into a single cluster.
    """

    return .5 * np.square(np.dot(weights, np.mean(X, axis=0)))


def objective(weights, X, a, b):
    """
    Objective function to be minimized.

    The objective function embeds the data X in 1D space and computes
    on average how well the data is separated into two evenly sized
    clusters located at x=±1.
    """

    embedding = get_embedding(weights, X[:])  # do I need the [:]?
    discriminant = get_discriminant(embedding, a, b)
    penalty = get_penalty(weights, X)
    score = np.mean(discriminant + penalty)

    return score
