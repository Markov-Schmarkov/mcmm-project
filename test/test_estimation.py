from mcmm import estimation as est, analysis as ana
import numpy as np
import random
import unittest
from nose.tools import assert_true, assert_false, assert_equals, assert_raises
from numpy.testing import assert_array_equal

def test_compute_counting_matrix():
    #check simple example with 10 states and 3 clusters
    a = np.array([1,2,1,2,2,1,1,1,2,1])
    matrix = est.Estimator(a, 3, 2).count_matrix
    assert_array_equal(np.array([[0,0,0],[0,2,1],[0,1,0]]), matrix)
    #check simple example with 10 states and 3 clusters, different lag_time
    a = np.array([1,2,1,2,2,1,1,1,2,1])
    matrix = est.Estimator(a, 1, 1).count_matrix
    assert_array_equal(np.array([[0,0,0],[0,2,3],[0,3,1]]), matrix)
    #check simple example with 10 states and 2 clusters
    a = np.array([0,1,0,1,1,0,0,0,1,0])
    matrix = est.Estimator(a, 1, 1).count_matrix
    assert_array_equal(np.array([[2,3],[3,1]]), matrix)
    #check simple example with 10 states and 2 clusters, maximal lag_time
    a = np.array([0,1,0,1,1,0,0,0,1,0])
    matrix = est.Estimator(a, 9, 1).count_matrix
    assert_array_equal(np.array([[1,0],[0,0]]), matrix)
    #check simple example with 10 states and 2 clusters, high window shift
    a = np.array([0,1,0,1,1,0,0,0,1,0])
    matrix = est.Estimator(a, 4, 6).count_matrix
    assert_array_equal(np.array([[0,1],[0,0]]), matrix)


def test_simple_markov():
    """This test generates data based on a small Markov model
    and tests the resulting transition matrix against the original transition matrix
    """
    A = np.array([[0.5,0.3,0.2],[0.2,0.6,0.2],[0.1,0.05,0.85]])
    n = 50000
    cluster_labels = np.zeros(n)
    cluster_labels[0] = 1;
    for i in range(1,n):
        zahl = np.random.rand(1)
        if zahl < A[cluster_labels[i-1],0]:
            cluster_labels[i] = 0
        elif zahl < A[cluster_labels[i-1],0] + A[cluster_labels[i-1],1]:
            cluster_labels[i] = 1
        else:
            cluster_labels[i] = 2
    estimator = est.Estimator(cluster_labels, 1, 1)
    np.testing.assert_allclose(estimator.transition_matrix, A, atol=0.05, rtol=0.1)


def test_transition_matrix():
    clusters = 10
    traj = np.random.randint(0, clusters, 50)
    transition_matrix = est.Estimator(traj, 1, 1).transition_matrix
    row_sums = transition_matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1)


def test_transition_matrix_reversible():
    traj = np.random.randint(0, 10, 50)
    transition_matrix = est.Estimator(traj, 1, 1).reversible_transition_matrix
    msm = ana.MarkovStateModel(transition_matrix)
    assert_true(msm.is_reversible)

