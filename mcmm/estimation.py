r"""
This module should handle the transition counting and transition matrix estimation.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np

def transition_matrix_from_count_matrix(count_matrix):
    row_sums = count_matrix.sum(axis=1)
    count_matrix /= row_sums[:, np.newaxis]
    return count_matrix


def reversible_transition_matrix_from_count_matrix(count_matrix):
    raise NotImplementedError()
