from mcmm import estimation as est
import numpy as np
import random
import unittest
from nose.tools import assert_true, assert_false, assert_equals, assert_raises

def test_transition_matrix():
    count_matrix = np.random.randint(0, 1000, (10, 10))
    print(count_matrix)
    transition_matrix = est.transition_matrix_from_count_matrix(count_matrix)
    print(transition_matrix)
    row_sums = transition_matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1)
    for i in range(len(count_matrix)):
        row_sum = sum(count_matrix[i,:])
        for j in range(len(count_matrix)):
            np.testing.assert_allclose(transition_matrix[i,j], count_matrix[i,j]/row_sum)
