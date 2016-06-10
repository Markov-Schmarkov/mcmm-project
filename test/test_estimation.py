from mcmm import estimation as est
import numpy as np
import random
import unittest
from nose.tools import assert_true, assert_false, assert_equals, assert_raises
from numpy.testing import assert_array_equal

def test_compute_counting_matrix():
    #check simple example with 10 states and 3 clusters
    a = np.array([1,2,1,2,2,1,1,1,2,1])
    matrix = est.count_matrix_from_cluster_labels_using_sliding_window(a, 3, 2, 3)
    assert_array_equal(np.array([[0,0,0],[0,2,1],[0,1,0]]), matrix)
    #check simple example with 10 states and 3 clusters, different lag_time
    a = np.array([1,2,1,2,2,1,1,1,2,1])
    matrix = est.count_matrix_from_cluster_labels_using_sliding_window(a, 1, 1, 3)
    assert_array_equal(np.array([[0,0,0],[0,2,3],[0,3,1]]), matrix)
    #check simple example with 10 states and 2 clusters
    a = np.array([0,1,0,1,1,0,0,0,1,0])
    matrix = est.count_matrix_from_cluster_labels_using_sliding_window(a, 1, 1, 2)
    assert_array_equal(np.array([[2,3],[3,1]]), matrix)
    #check simple example with 10 states and 2 clusters, maximal lag_time
    a = np.array([0,1,0,1,1,0,0,0,1,0])
    matrix = est.count_matrix_from_cluster_labels_using_sliding_window(a, 9, 1, 2)
    assert_array_equal(np.array([[1,0],[0,0]]), matrix)
    #check simple example with 10 states and 2 clusters, high window shift
    a = np.array([0,1,0,1,1,0,0,0,1,0])
    matrix = est.count_matrix_from_cluster_labels_using_sliding_window(a, 4, 6, 2)
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
    count_matrix = est.count_matrix_from_cluster_labels_using_sliding_window(cluster_labels,1,1,3)
    transition_matrix = est.transition_matrix_from_count_matrix(count_matrix)
    assert_equals(int(50*np.linalg.norm(transition_matrix - A)),0)

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
