r"""
This module should handle the analysis of an estimated Markov state model.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from .common import *

import numpy as np
import msmtools.analysis
import pandas as pd

class CommunicationClass:
    def __init__(self, states, closed):
        self.states = states
        self.closed = closed
        

class MarkovStateModel:

    def __init__(self, transition_matrix):
        """Create new Markov State Model.

        Parameters:
        transition_matrix: pandas.DataFrame where entry (a,b) contains transition probability a -> b
        """
        if not self.is_stochastic_matrix(transition_matrix):
            raise InvalidValue('Transition matrix must be stochastic')
        if not transition_matrix.shape[0] == transition_matrix.shape[1]:
            raise InvalidValue('Transition matrix must be quadratic')
        self._transition_matrix = transition_matrix
        self._backward_transition_matrix = None
        self._stationary_distribution = None
        self._num_states = transition_matrix.shape[0]
        self._is_aperiodic = None
        self._eigenvalues = None
        self._left_eigenvectors = None
        self._communication_classes = None

    @property
    def communication_classes(self):
        """The set of communication classes of the state space.
        Returns: [CommunicationClass]
            List of communication classes sorted by size descending.
        """
        if self._communication_classes is None:
            self._communication_classes = [
                CommunicationClass(sorted(c), component_is_closed(c, self.transition_matrix))
                for c in strongly_connected_components(self.transition_matrix)
            ]
        self._communication_classes.sort(key=lambda c: len(c.states), reverse=True)
        return self._communication_classes
    
    @property
    def is_irreducible(self):
        """Whether the markov chain is irreducible."""
        return (len(self.communication_classes) == 1)
    
    @property
    def is_aperiodic(self):
        """Whether the markov chain is aperiodic."""
        if self._is_aperiodic is None:
            self._is_aperiodic = self._determine_aperiodicity()
        return self._is_aperiodic
    
    def _determine_aperiodicity(self):
        period = -1
        irred = self.is_irreducible                                     # remember, if chain is irreducible
        for s in range (0, self._num_states):                           # we check period for all states
            pos = np.zeros(self._num_states)
            pos[s] = 1                                                  # we are only in state s right at the start
            for i in range(1, 2*self._num_states):                      # we need to check all paths of length <= 2|S| - 1
                pos = pos.dot(self._transition_matrix)                     # propagate
                pos[:] = pos[:] > 0                                     # normalize to avoid too small entries
                if pos[s] == 1:
                    period = gcd(i, period) if not period == -1 else i  # period of this state = gcd of all path lengths
                if period == 1:
                    if irred:                                           # irreducible chains with one state with period == 1 are aperiodic
                        return True
                    break
            if not period == 1:                                         # if there is a state with period > 1, chain is not aperiodic
                return False
            else:
                period = -1
        return True

    @property
    def transition_matrix(self):
        """The transition matrix where entry (a,b) denotes transition probability a->b"""
        return self._transition_matrix
    
    @property
    def backward_transition_matrix(self):
        if self._backward_transition_matrix is None:
            pi = self.stationary_distribution
            self._backward_transition_matrix = self.transition_matrix.T.mul(pi, axis=1).mul(1/pi, axis=0)
        return self._backward_transition_matrix
    
    @property
    def period(self):
        """Returns the period of the markov chain."""
        if not self.is_irreducible:
            raise InvalidOperation('Cannot compute period of reducible Markov chain')
        eigenvalues, _ = self._left_eigen()
        norms = np.absolute(eigenvalues)
        period = np.count_nonzero(np.isclose(norms, 1))
        assert(period >= 1)
        return period

    @property
    def stationary_distribution(self):
        if self._stationary_distribution is None:
            self._stationary_distribution = self._find_stationary_distribution()
        return self._stationary_distribution
       

    def left_eigenvector(self, k):
        """Computes the first k eigenvectors for largest eigenvalues
        
        Returns: eigenvalues    
        """
        number_of_eigenvectors = k +1
        ausgabe = np.zeros((len(self.transition_matrix),number_of_eigenvectors))
        eigenvalues, eigenvectors = self._left_eigen()
        for i in range (0,number_of_eigenvectors):
            maxi = 0
            jj = len(self.transition_matrix) + 10;
            for j in range (0,len(self.transition_matrix)):
                if(maxi < np.abs(eigenvalues[j])):
                    maxi = np.abs(eigenvalues[j])
                    jj = j
            if(jj < len(self.transition_matrix)+ 1):
                ausgabe[:,i] = eigenvectors[:,jj]
                eigenvalues[jj] = 0
        return ausgabe
    
    
    @property
    def is_reversible(self):
        """Whether the markov chain is reversible"""
        return np.allclose(self.backward_transition_matrix, self.transition_matrix)

    def _left_eigen(self):
        """Finds the eigenvalues and left eigenvectors of the transition matrix.
        
        Returns: (eigenvalues, eigenvectors)
            where eigenvalues[i] corresponds to eigenvectors[:,i]
        """
        if self._left_eigenvectors is None:
            self._eigenvalues, self._left_eigenvectors = np.linalg.eig(self.transition_matrix.T)
            self._left_eigenvectors = pd.DataFrame([
                pd.Series(x, index=self.transition_matrix.index)
                for x in self._left_eigenvectors
            ])
        return (self._eigenvalues, self._left_eigenvectors)

    def _right_eigen(self):
        """Finds the eigenvalues and right eigenvectors of the transition matrix.
        
        Returns: (eigenvalues, eigenvectors)
            where eigenvalues[i] corresponds to eigenvectors[:,i]
        """
        if not self._right_eigenvectors:
            self._eigenvalues, self._right_eigenvectors = np.linalg.eig(self.transition_matrix)
        return (self._eigenvalues, self._right_eigenvectors)
            
    def _find_stationary_distribution(self):
        """Finds the stationary distribution of a given stochastic matrix.
        The matrix is assumed to be irreducible.
        """
        if not self.is_irreducible:
            raise InvalidOperation('Cannot compute stationary distribution of reducible Markov chain')
        eigenvalues, eigenvectors = self._left_eigen()
        v = eigenvectors.iloc[:,np.isclose(eigenvalues, 1)].squeeze()
        assert(len(v.shape) == 1)
        v_real = v.apply(np.real)
        assert(np.allclose(v, v_real)) # result should be real
        return v_real/np.sum(v_real)
    
    def forward_committors(self, A, B):
        """Returns the vector of forward commitors from A to B"""
        return self._commitors(A, B, self.transition_matrix)
    
    def backward_commitors(self, A, B):
        """Returns the vector of backward commitors from A to B"""
        return self._commitors(B, A, self.backward_transition_matrix)

    def probability_current(self, A, B):
        """Returns the probability current from A to B.

        Returns:
        (n, n) ndarray containing the probabilty currents for every pair of states.
        """
        result = np.zeros(self.transition_matrix.shape)
        fwd_commitors = self.forward_committors(A, B)
        bwd_commitors = self.backward_commitors(A, B)
        for (i,j), value in np.ndenumerate(self.transition_matrix):
            if i != j:
                result[i,j] = self.stationary_distribution[i] * bwd_commitors[i] * value * fwd_commitors[j]
        return result

    def effective_probability_current(self, A, B):
        """Returns the effective probabiltiy current from A to B.

        Returns:
        (n, n) ndarray containing the effective probabilty currents for every pair of states.
        """
        current = self.probability_current(A, B)
        result = np.zeros(current.shape)
        for (i,j), value in np.ndenumerate(current):
            result[i,j] = max(0, current[i,j]-current[j,i])
        return result

    def transition_rate(self, A, B):
        """Returns the transition rate from A to B"""
        current = self.probability_current(A, B)
        num_trajs = np.sum(current[A,:])
        result = num_trajs / self.stationary_distribution.dot(self.backward_commitors(A,B))
        return result

    def mean_first_passage_time(self, A, B):
        """Returns the mean first-passage-time from A to B"""
        return 1/self.transition_rate(A, B)

    def pcca(self, num_sets):
        """Compute meta-stable sets using PCCA++ and return the membership of all states to these sets.

        Arguments:
        num_sets: integer
            Number of metastable sets

        Returns:
        clusters : (n, m) ndarray
            Membership vectors. clusters[i, j] contains the membership of state i to metastable state j.
        """
        return msmtools.analysis.pcca(self.transition_matrix, num_sets)
    
    def restriction(self, communication_class):
        """Returns the restriction of the model to a single communication class.
        
        Arguments:
        communication_class: CommunicationClass
            The model’s communication classes that the result should be restricted to. Required to be closed.
        
        Returns: MarkovStateModel
            The restricted markov chain. Note that the states will be re-indexed to range [0, n]
        """
        assert(communication_class.closed)
        return type(self)(self.transition_matrix.iloc[communication_class.states, communication_class.states])

    @staticmethod
    def _commitors(A, B, T):
        """Returns the vector of forward commitors from A to B given propagator T"""
        n = len(T)
        C = list(set(range(n)) - set().union(A, B))
        if C:
            M = T - np.identity(n)
            d = np.sum(M.iloc[C,B], axis=1)
            solution = np.linalg.solve(M.iloc[C, C], -d)
        result = np.empty(n)
        c = 0
        for i in range(n):
            if i in A:
                result[i] = 0
            elif i in B:
                result[i] = 1
            else:
                result[i] = np.real(solution[c])
                assert(np.isclose(result[i], solution[c])) # solution should be real
                c += 1
        return result

    @staticmethod
    def is_stochastic_matrix(A):
        return np.all(0 <= A) and np.all(A <= 1) and np.allclose(np.sum(A, axis=1), 1)
    

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
        if adjacency_matrix.iat[root, vertex]:
            if not flags[vertex]:
                result += depth_first_search(adjacency_matrix, vertex, flags)
    result.append(root)
    return result


def component_is_closed(component, adjacency_matrix):
    """Returns whether a component is closed, i.e. whether there are no
    edges pointing out of the component
    """
    for a in component:
        for b in set(range(adjacency_matrix.shape[0])).difference(component):
            if adjacency_matrix.iat[a,b] > 0:
                return False
    return True


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
    flags = [False] * len(nodes)
    components = []
    for node in reversed(node_list):
        if not flags[node]:
            components.append(depth_first_search(adjacency_matrix.T, node, flags))
    return components
    
def gcd(a, b):
    while b != 0:
        b, a = a%b, b
    return a
