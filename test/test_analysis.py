from mcmm import analysis as ana
import numpy as np
from nose.tools import assert_true

def generate_random_stochastic_matrix():
    matrix = np.random.rand(4,4)+0.001
    for i, row in enumerate(matrix):
        matrix[i] = row/sum(row)
    return matrix


def test_find_stationary_distribution():
    matrix = generate_random_stochastic_matrix()
    msm = ana.MarkovStateModel(matrix)
    distrib = msm.stationary_distribution
    np.testing.assert_array_almost_equal(np.dot(matrix.T, distrib), distrib)
    np.testing.assert_almost_equal(np.linalg.norm(distrib), 1)

