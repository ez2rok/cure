from scipy.optimize import minimize
import numpy as np
from icecream import ic

# import local files
from objective import objective


def minimize_obj(X_train, a=1.1, b=2, n_starts=10, record_history=True, random_state=None):
    """
    Minimize the objective function.

    ---- Parameters ----
    X_train (n_samples, n_features): training data
    n_starts (int): number of times to run the optimization, each with a different initial guess
    record_history (bool): record the weights at every iteration. This is a time
                                  consuming operation so it is not recommended to record but
                                  may be useful for plotting.
    random_state (int): random seed for the optimization. Because we are running the minimization
                        multiple times each with a different initial guess, we use the random state
                        to generate the n_start seeds that we will use to initialize the weights.
    """

    # initial values
    best_score = np.inf
    best_weights = None
    best_weight_history = None
    callback = (lambda xk: weight_history.append(xk)) if record_history else None
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**32, size=n_starts)

    for i in range(n_starts):

        # get weights that best minminimize the objective function
        weight_history = []
        weights_0 = np.random.default_rng(seeds[i]).normal(size=X_train.shape[1])
        res = minimize(objective, weights_0, args=(X_train, a, b),
                       callback=callback)

        # record weights with lower score
        score = objective(res.x, X_train, a, b)
        if score < best_score:
            best_score = score
            best_weights = res.x
            best_weight_history = np.vstack(
                weight_history) if record_history else None

    return best_weights, best_weight_history
