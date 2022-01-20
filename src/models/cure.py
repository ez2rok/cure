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

    def fit(self, X, y=None, record_history=False):
        """
        Minimize the loss function to find the weights that best separate 
        the data into two clusters.

        Parameters
        ----------
        X : (n_samples, n_features) array
            The data.
        y : None, optional
            Ignored. This parameter exists only for compatibility with Pipeline.
        record_history : bool, optional
            Record the weights at every iteration. This is a time consuming 
            operation so it is not recommended except for plotting. By default False.

        Parameters
        -------
        best_weights : (n_features,) array
            The weights that best minimize the loss function.
        best_weights_history : (n_iterations, n_features) array
            The weights at every iteration. If record_history is False, this
            will be None.

        Returns
        -------
        self : CURE
            The model with updated parameters.
        """

        # initial values
        weight_history = []
        callback = (lambda xk: weight_history.append(xk)) \
            if record_history else None
        rng = np.random.default_rng(self.random_state)

        # get weights that best mininimize the loss function
        weights_0 = rng.normal(size=X.shape[1])
        res = minimize(loss, weights_0, args=(X, self.a, self.b),
                       callback=callback)

        # record values
        self.weights = res.x
        self.weight_history = np.vstack(
            weight_history) if record_history else None
        self.score = loss(res.x, X, self.a, self.b)

        return self

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

    def set_params(self, **params):
        """
        Set the parameters of the model.

        Parameters
        ----------
        params : dict
            The parameters to set.

        Returns
        -------
        self : CURE
            The model with updated parameters.
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self
