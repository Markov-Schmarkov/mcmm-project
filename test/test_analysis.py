from mcmm import analysis as ana
import numpy as np
import random
import unittest
from nose.tools import assert_true, assert_false, assert_equals, assert_raises


def make_stochastic(matrix):
    for i, row in enumerate(matrix):
        matrix[i] = row/sum(row)
    return matrix


def test_find_stationary_distribution():
    matrix = np.random.rand(4, 4) + 0.001
    msm = ana.MarkovStateModel(make_stochastic(matrix))
    distrib = msm.stationary_distribution
    np.testing.assert_allclose(matrix.T.dot(distrib), distrib)
    np.testing.assert_allclose(np.sum(distrib), 1)


def test_find_stationary_distribution_periodic():
    matrix = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    np.testing.assert_allclose(msm.stationary_distribution, [1/3, 1/3, 1/3])


def test_find_stationary_distribution_raises():
    matrix = np.identity(3)
    msm = ana.MarkovStateModel(matrix)
    with assert_raises(ana.InvalidOperation):
        distrib = msm.stationary_distribution


def test_depth_first_search():
    nodes = 10
    matrix = np.random.randint(2, size=(nodes,nodes))
    root = random.randrange(0, nodes)
    flags = [False]*nodes
    result = ana.depth_first_search(matrix, root, flags)
    assert_equals(root, result[-1])
    for v1 in result:
        for v2 in set(range(10)) - set(result):
            assert_equals(0, matrix[v1, v2])


def test_strongly_connected_components():
    nodes = 10
    matrix = np.random.randint(2, size=(nodes,nodes))
    components = ana.strongly_connected_components(matrix)
    for c1, c2 in zip(components, components[1:]):
        for v1 in c1:
            for v2 in c2:
                assert_true(matrix[v1,v2] == 0 or matrix[v2,v1] == 0)
    assert_equals(set(range(nodes)), set().union(*components))


def test_forward_commitor():
    matrix = np.random.rand(4,4)
    matrix[0,1] = matrix[0,2] + matrix[0,3]
    msm = ana.MarkovStateModel(make_stochastic(matrix))
    np.testing.assert_array_almost_equal(msm.forward_committors([1], [2, 3]), [0.5, 0, 1, 1])


def test_backward_commitor():
    matrix = np.random.rand(3,3)
    matrix[1,2] = matrix[2,1]
    matrix[1,1] = matrix[2,2]
    matrix[1,0] = matrix[2,0]
    matrix[0,1] = matrix[0,2]
    msm = ana.MarkovStateModel(make_stochastic(matrix))
    np.testing.assert_array_almost_equal(msm.backward_commitors([1], [2]), [0.5, 1, 0])


def test_reversible():
    matrix = np.array([
        [ 0.9,  0.1,    0,    0],
        [ 0.1, 0.89, 0.01,    0],
        [   0, 0.01, 0.79,  0.2],
        [   0,    0,  0.2,  0.8]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_true(msm.is_reversible)


def test_not_reversible():
    matrix = np.array([
        [ 0.9,  0.1,    0],
        [   0,  0.9,  0.1],
        [ 0.1,    0,  0.9]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_false(msm.is_reversible)


def test_periodic():
    matrix = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_equals(msm.period, 3)
    assert_false(msm.is_aperiodic)


def test_aperiodic():
    matrix = np.random.rand(4, 4) + 0.001
    matrix[0,0] = 0.1
    msm = ana.MarkovStateModel(make_stochastic(matrix))
    assert_equals(msm.period, 1)
    assert_true(msm.is_aperiodic)


# Periodicity tests
# These should be aperiodic
def test_aperiodic_normal():
    matrix = np.array([
        [ 0.9,  0.1,    0,    0],
        [ 0.1, 0.89, 0.01,    0],
        [   0, 0.01, 0.79,  0.2],
        [   0,    0,  0.2,  0.8]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_true(msm.is_aperiodic)

def test_aperiodic_gcd():
    matrix = np.array([
        [ 0, 1, 0, 0],
        [ 0.1, 0, 0.9, 0],
        [ 0, 0, 0, 1],
        [ 0, 1, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_true(msm.is_aperiodic)

def test_aperiodic_path():
    matrix = np.array([
        [0, 1, 0, 0, 0],
        [0.5, 0, 0.5, 0, 0],
        [0, 0.5, 0, 0.5, 0],
        [0, 0, 0.5, 0, 0.5],
        [0, 0, 0, 0.9, 0.1]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_true(msm.is_aperiodic)
    
def test_one_self_returning_state():
    matrix = np.array([
        [0.5, 0.5, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_true(msm.is_aperiodic)
    
def test_one_self_returning_state_reducible():
    matrix = np.array([
        [0.5, 0.5, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.5, 0.5],
        [0, 0, 0, 0, 0.5, 0.5]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_true(msm.is_aperiodic)

# These should be periodic
def test_periodic_circle():
    matrix = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_false(msm.is_aperiodic)
    
def test_periodic_path():
    matrix = np.array([
        [0, 1, 0, 0, 0],
        [0.5, 0, 0.5, 0, 0],
        [0, 0.5, 0, 0.5, 0],
        [0, 0, 0.5, 0, 0.5],
        [0, 0, 0, 1, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_false(msm.is_aperiodic)
    
def test_periodic_reducible():
    matrix = np.array([
        [0.5, 0.5, 0, 0, 0],
        [0.5, 0.5, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_false(msm.is_aperiodic)

def test_transient_state():
    matrix = np.array([
        [0, 0.5, 0.5, 0, 0],
        [0, 0.5, 0.5, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_false(msm.is_aperiodic)
# End periodicity tests

