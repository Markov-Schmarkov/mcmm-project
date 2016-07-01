r"""
This module should handle the transition counting and transition matrix estimation.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

from .common import *

import numpy as np
import pandas as pd

class Estimator:
    def __init__(self, trajectories, lag_time=1, window_shift=1):
        """Constructor.

        Arguments:
        trajectory: 1-dimensional np.ndarray
            Entry #i contains the cluster number at time i. Cluster numbers should be non-negative integers.
        lag_time: int, default=1
            Lag time of the markov process
        window_shift: int, default=1
            Window shifting distance of the estimator. Value should be in range 1 to lag_time.
        """
        if isinstance(trajectories, np.ndarray) and len(trajectories.shape) == 1:
            trajectories = [trajectories]
        self._lag_time = lag_time
        self._window_shift = window_shift
        self._num_clusters = max(np.max(t) for t in trajectories)+1

        self._count_matrix = np.zeros((self._num_clusters,self._num_clusters), dtype=np.dtype(int))

        for traj in trajectories:
            self._update_count_matrix(traj)

        self._transition_matrix = None
        self._reversible_transition_matrix = None

    @property
    def count_matrix(self):
        return pd.DataFrame(self._count_matrix)

    def _update_count_matrix(self, traj):
        """Updates the count matrix by adding the transitions occuring in the given trajectory. The method is the sliding window approach."""
        length = traj.shape[0]
        for s in range(0, length-self._lag_time, self._window_shift):
            self._count_matrix[traj[s], traj[s + self._lag_time]] += 1

    @property
    def _np_transition_matrix(self):
        if self._transition_matrix  is None:
            self._transition_matrix = make_stochastic(self._count_matrix)
        return self._transition_matrix
    
    @property
    def transition_matrix(self):
        return pd.DataFrame(self._np_transition_matrix)

    @property
    def reversible_transition_matrix(self):
        if self._reversible_transition_matrix is None:
            self._reversible_transition_matrix = self._compute_reversible_transition_matrix()
        return pd.DataFrame(self._reversible_transition_matrix)

    def _compute_reversible_transition_matrix(self):
        matrix = self._np_transition_matrix
        matrix_next = np.zeros(matrix.shape)
        while not np.allclose(matrix, matrix_next, rtol=0.01):
            for i,j in np.ndindex(*self._count_matrix.shape):
                d_i = self._count_matrix[i,:].sum() / matrix[i,:].sum()
                d_j = self._count_matrix[j,:].sum() / matrix[j,:].sum()
                matrix_next[i,j] = (self._count_matrix[i,j] + self._count_matrix[j,i])/(d_i + d_j)
            matrix, matrix_next = matrix_next, matrix
        return make_stochastic(matrix)


def make_stochastic(matrix):
    row_sums = matrix.sum(axis=1)
    if not np.all(row_sums > 0):
        raise InvalidValue('Input matrix contains all-zero rows.')
    return matrix / row_sums[:, np.newaxis]

