from mcmm import analysis as ana, estimation as est
import numpy as np
import pandas as pd
import random
import unittest
from nose.tools import assert_true, assert_false, assert_equals, assert_raises
import pandas.util.testing as pdt


def make_stochastic(matrix):
    for i in matrix:
        matrix.iloc[i] /= sum(matrix.iloc[i])
    return matrix


def test_find_stationary_distribution():
    matrix = pd.DataFrame(np.random.rand(4, 4) + 0.001)
    msm = ana.MarkovStateModel(make_stochastic(matrix))
    distrib = msm.stationary_distribution
    np.testing.assert_allclose(matrix.T.dot(distrib), distrib)
    np.testing.assert_allclose(np.sum(distrib), 1)


def test_find_stationary_distribution_periodic():
    matrix = pd.DataFrame([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    np.testing.assert_allclose(msm.stationary_distribution, [1/3, 1/3, 1/3])


def test_find_stationary_distribution_raises():
    matrix = pd.DataFrame(np.identity(3))
    msm = ana.MarkovStateModel(matrix)
    with assert_raises(ana.InvalidOperation):
        distrib = msm.stationary_distribution


def test_depth_first_search():
    nodes = 10
    matrix = pd.DataFrame(np.random.randint(2, size=(nodes,nodes)))
    root = random.randrange(0, nodes)
    flags = [False]*nodes
    result = ana.depth_first_search(matrix, root, flags)
    assert_equals(root, result[-1])
    for v1 in result:
        for v2 in set(range(10)) - set(result):
            assert_equals(0, matrix[v1, v2])


def test_strongly_connected_components():
    nodes = 10
    matrix = pd.DataFrame(np.random.randint(2, size=(nodes,nodes)))
    components = ana.strongly_connected_components(matrix)
    for c1, c2 in zip(components, components[1:]):
        for v1 in c1:
            for v2 in c2:
                assert_true(matrix[v1,v2] == 0 or matrix[v2,v1] == 0)
    assert_equals(set(range(nodes)), set().union(*components))


def test_forward_commitor():
    matrix = pd.DataFrame(np.random.rand(4,4))
    matrix.iat[0,1] = matrix.iat[0,2] + matrix.iat[0,3]
    msm = ana.MarkovStateModel(make_stochastic(matrix))
    np.testing.assert_array_almost_equal(msm.forward_committors([1], [2, 3]), [0.5, 0, 1, 1])


def test_backward_commitor():
    matrix = pd.DataFrame(np.random.rand(3,3))
    matrix.iat[1,2] = matrix.iat[2,1]
    matrix.iat[1,1] = matrix.iat[2,2]
    matrix.iat[1,0] = matrix.iat[2,0]
    matrix.iat[0,1] = matrix.iat[0,2]
    msm = ana.MarkovStateModel(make_stochastic(matrix))
    np.testing.assert_array_almost_equal(msm.backward_commitors([1], [2]), [0.5, 1, 0])


def test_commitor_edgecase():
    """Test if commitors work if A union B is everything"""
    matrix = pd.DataFrame(np.random.rand(4, 4) + 0.001)
    msm = ana.MarkovStateModel(make_stochastic(matrix))
    np.testing.assert_array_almost_equal(msm.forward_committors([0, 1], [2, 3]), [0, 0, 1, 1])

def test_reversible():
    matrix = pd.DataFrame([
        [ 0.9,  0.1,    0,    0],
        [ 0.1, 0.89, 0.01,    0],
        [   0, 0.01, 0.79,  0.2],
        [   0,    0,  0.2,  0.8]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_true(msm.is_reversible)


def test_not_reversible():
    matrix = pd.DataFrame([
        [ 0.9,  0.1,    0],
        [   0,  0.9,  0.1],
        [ 0.1,    0,  0.9]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_false(msm.is_reversible)


def test_periodic():
    matrix = pd.DataFrame([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_equals(msm.period, 3)
    assert_false(msm.is_aperiodic)


def test_aperiodic():
    matrix = pd.DataFrame(np.random.rand(4, 4) + 0.001)
    matrix.iat[0,0] = 0.1
    msm = ana.MarkovStateModel(make_stochastic(matrix))
    assert_equals(msm.period, 1)
    assert_true(msm.is_aperiodic)


# Periodicity tests
# These should be aperiodic
def test_aperiodic_normal():
    matrix = pd.DataFrame([
        [ 0.9,  0.1,    0,    0],
        [ 0.1, 0.89, 0.01,    0],
        [   0, 0.01, 0.79,  0.2],
        [   0,    0,  0.2,  0.8]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_true(msm.is_aperiodic)

def test_aperiodic_gcd():
    matrix = pd.DataFrame([
        [ 0, 1, 0, 0],
        [ 0.1, 0, 0.9, 0],
        [ 0, 0, 0, 1],
        [ 0, 1, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_true(msm.is_aperiodic)

def test_aperiodic_path():
    matrix = pd.DataFrame([
        [0, 1, 0, 0, 0],
        [0.5, 0, 0.5, 0, 0],
        [0, 0.5, 0, 0.5, 0],
        [0, 0, 0.5, 0, 0.5],
        [0, 0, 0, 0.9, 0.1]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_true(msm.is_aperiodic)
    
def test_one_self_returning_state():
    matrix = pd.DataFrame([
        [0.5, 0.5, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_true(msm.is_aperiodic)
    
def test_one_self_returning_state_reducible():
    matrix = pd.DataFrame([
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
    matrix = pd.DataFrame([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_false(msm.is_aperiodic)
    
def test_periodic_path():
    matrix = pd.DataFrame([
        [0, 1, 0, 0, 0],
        [0.5, 0, 0.5, 0, 0],
        [0, 0.5, 0, 0.5, 0],
        [0, 0, 0.5, 0, 0.5],
        [0, 0, 0, 1, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_false(msm.is_aperiodic)
    
def test_periodic_reducible():
    matrix = pd.DataFrame([
        [0.5, 0.5, 0, 0, 0],
        [0.5, 0.5, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_false(msm.is_aperiodic)

def test_transient_state():
    matrix = pd.DataFrame([
        [0, 0.5, 0.5, 0, 0],
        [0, 0.5, 0.5, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0]
    ])
    msm = ana.MarkovStateModel(matrix)
    assert_false(msm.is_aperiodic)
# End periodicity tests

def test_pcca():
    num_states = 10
    num_clusters = 4
    traj = np.random.randint(0, num_states, 100)
    transition_matrix = est.Estimator(traj).reversible_transition_matrix
    msm = ana.MarkovStateModel(transition_matrix)
    result = msm.pcca(num_clusters)
    assert_equals(result.shape, (num_states, num_clusters))
    assert_true(np.all(result <= 1) and np.all(result >= 0))


def test_transition_rate():
    matrix = pd.DataFrame(np.random.rand(4, 4) + 0.001)
    msm = ana.MarkovStateModel(make_stochastic(matrix))
    rate = msm.transition_rate([1], [2, 3])
    assert_true(rate > 0)


def test_restriction():
    matrix = pd.DataFrame([
        [0.4, 0.2, 0.2, 0.2],
        [  0, 0.4, 0.5, 0.1],
        [  0, 0.1, 0.7, 0.2],
        [  0, 0.6, 0.1, 0.3]
    ])
    msm = ana.MarkovStateModel(matrix)
    classes = msm.communication_classes
    assert_equals(len(classes), 2)
    assert_false(classes[1].closed)
    assert_true(classes[0].closed)
    assert_equals(classes[1].states, [0])
    assert_equals(classes[0].states, [1, 2, 3])
    msm2 = msm.restriction(classes[0])
    print(msm2.transition_matrix)
    print(matrix)
    print(msm.transition_matrix)
    pdt.assert_frame_equal(msm2.transition_matrix, pd.DataFrame.from_items({
        1: [0.4, 0.5, 0.1],
        2: [0.1, 0.7, 0.2],
        3: [0.6, 0.1, 0.3]
    }.items(), orient='index', columns=[1, 2, 3]))

