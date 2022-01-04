from scipy.optimize import minimize
import numpy as np
from icecream import ic

# import local files
from .loss import loss, get_embedding


class CURE:
    """
    CURE clustering class.
    """

    def __init__(self, a=1.1, b=2, random_state=None):
        self.a = a
        self.b = b
        self.random_state = random_state

    def fit(self, X, y=None, n_starts=3, record_history=False):
        """
        Minimize the loss function to find the weights that best separate 
        the data into two clusters.

        Each time we minmize the loss function it is with different initial weights.
        So we use self.random_state to genreate n_starts different seeds. Each
        seed seeds a random number generator from which the initial weights
        are drawn according to a normal distribution.

        Parameters
        ----------
        X : (n_samples, n_features) array
            The data.
        y : None, optional
            Ignored. This parameter exists only for compatibility with Pipeline.
        n_starts : int, optional
            Number of times to run the optimization, each with a different 
            initial weights, by default 3.
        record_history : bool, optional
            Record the weights at every iteration. This is a time consuming 
            operation so it is not recommended except for plotting. By default False.

        Returns
        -------
        best_weights : (n_features,) array
            The weights that best minimize the loss function.
        best_weights_history : (n_iterations, n_features) array
            The weights at every iteration. If record_history is False, this
            will be None.
        """

        # initial values
        best_score = np.inf
        best_weights = None
        best_weight_history = None
        callback = (lambda xk: weight_history.append(xk)) \
            if record_history else None
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**32, size=n_starts)

        for i in range(n_starts):

            # get weights that best minminimize the loss function
            weight_history = []
            weights_0 = np.random.default_rng(seeds[i]).normal(size=X.shape[1])
            res = minimize(loss, weights_0, args=(X, self.a, self.b),
                           callback=callback)

            # if there is a lower score, update the weights
            score = loss(res.x, X, self.a, self.b)
            if score < best_score:
                best_score = score
                best_weights = res.x
                best_weight_history = np.vstack(
                    weight_history) if record_history else None

        # record values
        self.weights = best_weights
        self.weight_history = best_weight_history
        self.score = best_score

        return best_weights, best_weight_history

    def predict(self, X, weights=None):
        """
        Predict the target of all samples of the data as a 1 or a -1.

        After computing the weights that best separate the data, embed
        the data in 1D and separate the positive and negative data into
        two different clusters.

        Parameters
        ----------
        X : (n_samples, n_features) array
            The data.
        weights : (n_features,) array, optional
            The weights used to embed the data in 1D. If None, self.weights
            are used. By default None.

        Returns
        -------
        y_pred : (n_samples,) array
            The predicted targets for the data.
        """

        weights = self.weights if weights is None else weights
        y_pred = np.sign(get_embedding(X, weights))
        return y_pred

    def fit_predict(self, X, **kwargs):
        """
        Fit the model and predict the target of all the samples in the data.

        Parameters
        ----------
        X : (n_samples, n_features) array
            The data.

        Returns
        -------
        y_pred : (n_samples,) array
            The predicted targets for the data.
        """

        self.fit(X, **kwargs)
        y_pred = self.predict(X)
        return y_pred
