import numpy as np
import sklearn.mixture
import multiprocessing

class GMM():

    def __init__(self, means, covariances, weights):
        """
        Gaussian Mixture Model Distribution class for calculation of log likelihood and sampling.

        Parameters
        ----------
        means : 2-D array_like of shape (n_mixtures, n_features)
            Means for each component of the GMM
        covariances : 2-D array_like of shape (n_mixtures, n_features)
            Covariance matrices of the GMM. Only diagonal matrices are supported at this time.
        weights : 1-D array_like of shape (n_mixtures,)
            Weights for each of the GMM components
        """
        if len(covariances.shape) == 2:
            self.covariance_type = 'diag'
        else:
            raise NotImplementedError('Only diagonal covariance matrices supported')
        self.gmm = sklearn.mixture.GaussianMixture(n_components=len(weights), covariance_type='diag')
        self.gmm.weights_ = weights
        self.gmm.covariances_ = covariances
        self.gmm.means_ = means
        self.gmm.precisions_cholesky_ = np.sqrt(1./covariances)
        self.n_mixtures = len(weights)
        try:
            self.n_features = means.shape[1]
        except:
            raise ValueError("Means array must be 2 dimensional")

    @property
    def means(self):
        return self.gmm.means_

    @property
    def covars(self):
        return self.gmm.covars_

    @property
    def weights(self):
        return self.gmm.weights_

    def sample(self, n_samples):
        """
        Sample from the GMM.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
            : 2-D array_like of shape (n_samples, n_features)
            Samples drawn from the GMM distribution
        """
        X, y = self.gmm.sample(n_samples)
        return X

    def log_likelihood(self, X, n_jobs=1):
        """
        Calculate the average log likelihood of the data given the GMM parameters

        Parameters
        ----------
        X : 2-D array_like of shape (n_samples, n_features)
            Data to be used.
        n_jobs : int
            Number of CPU cores to use in the calculation

        Returns
        -------
            : float
            average log likelihood of the data given the GMM parameters

        Notes
        -------
        For GMMs with small numbers of mixtures (<10) the use of more than 1 core can slow down the function.
        """
        return self.gmm.score_samples(X)
