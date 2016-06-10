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