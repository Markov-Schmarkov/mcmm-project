r"""
This module should handle the analysis of an estimated Markov state model.
"""

import numpy as np

class MarkovStateModel:

    def __init__(self, transition_matrix):
        self._transition_matrix = transition_matrix
        self._stationary_distribution = None

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def stationary_distribution(self):
        if self._stationary_distribution is None:
            self._stationary_distribution = self._find_stationary_distribution()
        return self._stationary_distribution

    def _find_stationary_distribution(self):
        """Finds the stationary distribution of a given stochastic matrix.
        The matrix is assumed to be irreducible.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        norms = [np.absolute(v) for v in eigenvalues]
        i = np.argmax(norms)
        assert(np.isclose(eigenvalues[i], 1))
        return eigenvectors[:,i]



