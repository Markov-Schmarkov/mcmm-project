r"""
This module should handle the analysis of an estimated Markov state model.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
import numpy as np

class Error(Exception):
    """Base class for all exceptions raised by the mcmm module."""

class InvalidOperation(Error):
    """An operation was called on a object that does not support it."""

class MarkovStateModel:

    def __init__(self, transition_matrix):
        """Create new Markov State Model.

        Parameters:
        transition_matrix: 2-dimensional numpy.ndarray where entry (a,b) contains transition probability a -> b
        """
        self._transition_matrix = transition_matrix
        self._stationary_distribution = None
        self._is_irreducible = None
        self._is_reversible = None

    @property
    def is_irreducible(self):
        """Whether the markov chain is irreducible."""
        if self._is_irreducible is None:
            self._is_irreducible = (len(strongly_connected_components(self.transition_matrix)) == 1)
        return self._is_irreducible

    @property
    def transition_matrix(self):
        """The transition matrix where entry (a,b) denotes transition probability a->b"""
        return self._transition_matrix

    @property
    def stationary_distribution(self):
        if self._stationary_distribution is None:
            self._stationary_distribution = self._find_stationary_distribution()
        return self._stationary_distribution

    @property
    def is_reversible(self):
        """Whether the markov chain is reversible"""
        if self._is_reversible is None:
            self._is_reversible = self._determine_reversibility()
        return self._is_reversible

    def _determine_reversibility(self):
        pi = self.stationary_distribution
        T = self.transition_matrix
        for i in range(len(T)):
            for j in range(len(T)):
                if not np.isclose(pi[i]*T[i,j], pi[j]*T[j,i]):
                    return False
        return True

    def _find_eigenvalues(self):
        """Finds the eigenvalues of a given stochastic matrix.
        The matrix is assumed to be irreducible.
        """
        if not self.is_irreducible:
            raise InvalidOperation('Cannot compute eigenvalues of reducible Markov chain')
        eigenvalues = np.linalg.eigvals(self.transition_matrix.T)
        print(eigenvalues)

    def _find_eigenvectors(self):
        """Find the eigenvectors of a given stochastic matrix.
        The matrix is assumed to be irreducible.
        """
        if not self.is_irreducible:
            raise InvalidOperation('Cannot compute eigenvalues of reducible Markov chain')
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        return eigenvectors

    def _find_stationary_distribution(self):
        """Finds the stationary distribution of a given stochastic matrix.
        The matrix is assumed to be irreducible.
        """
        if not self.is_irreducible:
            raise InvalidOperation('Cannot compute stationary distribution of reducible Markov chain')
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        norms = [np.absolute(v) for v in eigenvalues]
        i = np.argmax(norms)
        assert(np.isclose(eigenvalues[i], 1))
        return eigenvectors[:,i]


def depth_first_search(adjacency_matrix, root, flags):
    """Performs depth-first search on a digraph.
    
    Parameters:
    adjacency_matrix: numpy.ndarray of shape (n, n) containing node-node adjancencies.
    root: Root node
    flags: List of vertex flags. All vertices whose flag is initially set are ignored. After return the flags of all found vertices will be set.
    
    Returns a list of all nodes reachable from root sorted by
    post-order traversal.
    """
    result = []
    flags[root] = True
    for vertex in range(adjacency_matrix.shape[0]):
        if adjacency_matrix[root,vertex]:
            if not flags[vertex]:
                result += depth_first_search(adjacency_matrix, vertex, flags)
    result.append(root)
    return result


def strongly_connected_components(adjacency_matrix):
    """Finds all strongly connected components of a digraph.
    
    Parameters:
    adjacency_matrix: numpy.ndarray of shape (n, n) containing node-node adjancencies.
    
    Returns a list of strongly connected components, each of which is a list of vertices.
    """
    nodes = range(adjacency_matrix.shape[0])
    flags = [False] * len(nodes)
    node_list = []
    for node in nodes:
        if not flags[node]:
            node_list += depth_first_search(adjacency_matrix, node, flags)
    for n in nodes:
        assert(flags[n])
    flags = [False] * len(nodes)
    components = []
    for node in reversed(node_list):
        if not flags[node]:
            components.append(depth_first_search(adjacency_matrix.T, node, flags))
    return components
