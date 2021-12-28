import numpy as np
import matplotlib.pyplot as plt


def get_embedding(weights, X):
    """
    Embed the data into 1D space via the dot product.

    Parameters
    ----------
    weights : (n_features,) array
        The ith value controls the influence of the ith feature.
    X : (n_samples, n_features) array
        The data to be embedded.

    Returns
    -------
    (n_samples,) array
        The 1D embedding of the data.
    """
    return np.dot(weights, X.T)


def get_discriminant(X_embd, a, b):
    """
    Discriminative function that separates the embedded data X_embd into 
    two clusters.

    Because f() has a minimum at both x=±1, minimizing f() will map many of our
    datapoints to x=1 and many of them to x=-1, resulting in two different clusters.
    Because h() becomes huge for large x values, we construct f() which is just
    like h() except when x is too big we clip its growth with linear functions 
    and a cubic spline interpolation. 

    Note 1: 1 < a < b. 

    Note 2: 
    1.  if |x| <= a, f() has two valleys around ±1.
    2.  if a < |x| <= b, f() is a cubic spline connecting the valleys and the linear 
            function.
    3.  if |x| > b, f() is a linear function.

    Parameters
    ----------
    X_embd : (n_samples,) array
        The data embedding.
    a : float
        The threshold between h() and the cubic spline in the 
        discriminant function.
    b : float
        The threshold between the cubic spline and the linear function
        in the discriminant function.

    Returns
    -------
    (n_samples,) array
        The seperated (discriminated) 1D data embedding.
    """

    # initial values
    X_abs = np.abs(X_embd)
    X_result = np.empty(X_embd.shape)

    # define h = (x^2 - 1)^2 / 4 and its derivatives
    def h(x): return np.square(np.square(x) - 1) / 4
    def h_prime1(x): return x * (np.square(x) - 1)
    def h_prime2(x): return 3 * np.square(x) - 1

    # case 1: |x| <= a
    idxs = np.where(X_abs <= a)
    X_result[idxs] = h(X_embd[idxs])

    # case 2: a < |x| <= b
    idxs = np.where((a < X_abs) & (X_abs <= b))
    max_case1 = h(a)
    X_result[idxs] = max_case1 + h_prime1(a) * (X_abs[idxs] - a) \
        + h_prime2(a) * np.power(X_abs[idxs] - a, 2) / 2 \
        + h_prime2(a) * np.power(X_abs[idxs] - a, 3) / (6 * (b - a))

    # case 3: |x| > b
    idxs = np.where(X_abs > b)
    max_case2 = h(a) + h_prime1(a) * (b - a) \
        + h_prime2(a) * np.power(b - a, 2) / 2 \
        + h_prime2(a) * np.power(b - a, 3) / (6 * (b - a))
    X_result[idxs] = max_case2 \
        + (h_prime1(a) + 0.5 * (b - a) * h_prime2(a)) \
        * (X_abs[idxs] - b)

    return X_result


def plot_discriminant(outdir):
    """
    Plot the discriminant functions f() and h().

    Parameters
    ----------
    outdir : str
        The output directory.

    Returns
    -------
    matplotlib.figure.Figure
        The figure of the discriminant function.
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

    return fig


def get_penalty(weights, X):
    """
    Compute the penalty term.

    The penalty term encourages the data X to be evenly split between
    the two clusters at x=±1. This prevents the objective function 
    from achieving the trivial solution of clustering all of the
    data into a single cluster.

    Parameters
    ----------
    weights : (n_features,) array
        The ith value controls the influence of the ith feature.
    X : (n_samples, n_features) array
        The data.

    Returns
    -------
    float
        The penalty term that encourages the data to be evenly split between
        the two clusters at x=±1.
    """

    return .5 * np.square(np.dot(weights, np.mean(X, axis=0)))


def loss(weights, X, a, b):
    """
    Objective function to be minimized.

    The objective function embeds the data X in 1D space and computes
    on average how well the data is separated into two evenly sized
    clusters located at x=±1.

    Parameters
    ----------
    weights : (n_features,) array
        The ith value controls the influence of the ith feature.
        We will minimize this function by adjusting the weights.
    X : (n_samples, n_features) array
        The data.
    a : float
        The threshold between h() and the cubic spline in the 
        discriminant function.
    b : float
        The threshold between the cubic spline and the linear function 
        in the discriminant function.

    Returns
    -------
    score : float
        A measure of how well, on average, the data is separated into 
        two evenly sized clusters located at x=±1. The lower the score,
        the better.
    """

    embedding = get_embedding(weights, X[:])  # do I need the [:]?
    discriminant = get_discriminant(embedding, a, b)
    penalty = get_penalty(weights, X)
    score = np.mean(discriminant + penalty)

    return score
