r"""
This module should handle the transition counting and transition matrix estimation.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from .common import *

import numpy as np

class Estimator:
    def __init__(self, trajectory, lag_time=1, window_shift=1):
        """Constructor.

        Arguments:
        trajectory: 1-dimensional np.ndarray
            Entry #i contains the cluster number at time i. Cluster numbers should be non-negative integers.
        lag_time: int, default=1
            Lag time of the markov process
        window_shift: int, default=1
            Window shifting distance of the estimator. Value should be in range 1 to lag_time.
        """
        self._trajectory = trajectory
        self._lag_time = lag_time
        self._window_shift = window_shift
        self._num_clusters = np.max(trajectory)+1

        self._count_matrix = None
        self._transition_matrix = None
        self._reversible_transition_matrix = None
        
        for state in range(self._num_clusters):
            if state not in self._trajectory:
                raise InvalidValue('Data contains no transitions from state {}.'.format(state))

    @property
    def count_matrix(self):
        if self._count_matrix is None:
            self._count_matrix = self._compute_count_matrix()
        return self._count_matrix


    def _compute_count_matrix(self):
        """Computes the count matrix based on the cluster labels. The method is the sliding window approach."""
        num_states = self._trajectory.shape[0]
        count_matrix = np.zeros((self._num_clusters,self._num_clusters))
        for s in range(0, num_states-self._lag_time, self._window_shift):
            count_matrix[self._trajectory[s], self._trajectory[s + self._lag_time]] += 1
        return count_matrix

    @property
    def transition_matrix(self):
        if self._transition_matrix  is None:
            self._transition_matrix = make_stochastic(self.count_matrix)
        return self._transition_matrix

    @property
    def reversible_transition_matrix(self):
        if self._reversible_transition_matrix is None:
            self._reversible_transition_matrix = self._compute_reversible_transition_matrix()
        return self._reversible_transition_matrix

    def _compute_reversible_transition_matrix(self):
        matrix = self.transition_matrix
        matrix_next = np.zeros(matrix.shape)
        while not np.allclose(matrix, matrix_next, rtol=0.01):
            for i,j in np.ndindex(*self.count_matrix.shape):
                d_i = self.count_matrix[i,:].sum() / matrix[i,:].sum()
                d_j = self.count_matrix[j,:].sum() / matrix[j,:].sum()
                matrix_next[i,j] = (self.count_matrix[i,j] + self.count_matrix[j,i])/(d_i + d_j)
            matrix, matrix_next = matrix_next, matrix
        return make_stochastic(matrix)


def make_stochastic(matrix):
    row_sums = matrix.sum(axis=1)
    return matrix / row_sums[:, np.newaxis]
