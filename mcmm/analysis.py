r"""
This module should handle the analysis of an estimated Markov state model.
"""

import numpy as np


def find_stationary_distribution(matrix):
    """Finds the stationary distribution of a given stochastic matrix.
    The matrix is assumed to be irreducible.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    norms = [np.absolute(v) for v in eigenvalues]
    i = np.argmax(norms)
    assert(np.isclose(eigenvalues[i], 1))
    return eigenvectors[:,i]



