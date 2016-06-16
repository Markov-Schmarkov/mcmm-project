from mcmm import estimation as est, analysis as ana, clustering as cl
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
    
def test_clustering_estimation_simple_markov():
    """This test generates data based on a transition matrix. Then we do clustering.
    Then we check if the estimated matrix behaves as expected.
    """
    A = np.array([[0.5, 0.4, 0.1, 0],[0.2, 0.8, 0, 0],[0,0.05,0.25,0.7],[0,0,0.75,0.25]])
    for i, row in enumerate(A):
        A[i] = row/sum(row)
    n = 10000
    states = np.zeros((n,1))
    states[0,0] = 1
    noise = 1
    factor = 0.001
    for i in range(1,n):
        zahl = np.random.rand(1)
        if zahl < A[int(states[i-1,0]),0]:
            states[i,0] = 0 + factor * np.random.rand() /noise
        elif zahl < A[int(states[i-1,0]),0] + A[int(states[i-1,0]),1]:
            states[i,0] = 1 + factor * np.random.rand() /noise
        elif zahl < A[int(states[i-1,0]),0] + A[int(states[i-1,0]),1] + A[int(states[i-1,0]),2]:
            states[i,0] = 2 + factor * np.random.rand() /noise
        else:
            states[i,0] = 3 + factor * np.random.rand() /noise
    #do the clustering  
    clustering = cl.KMeans(states,4,method='kmeans++')
    cluster_centers = clustering.cluster_centers
    cluster_labels = clustering.cluster_labels
    cluster_labels = np.array(cluster_labels)
    #cluster_labels = np.zeros(n)
    #for i in range(1,n):
    #    cluster_labels[i] = cluster_labels[i]//1
    #do the estimation
    estimator = est.Estimator(cluster_labels, 1, 1)
    matrix = estimator.transition_matrix
    print(A)
    print(cluster_centers)
    print(matrix)
    """
    Q = np.zeros((4,4))
    for i in range(0,4):
        index = np.min(cluster_centers)
        Q[i,i] = 1
    Qt = Q
    print(Q)
    for i in range(0,4):
        index = np.argmin(cluster_centers)
        cluster_centers[index] = cluster_centers[index] + 10
        P = np.zeros((4,4))
        for j in range(0,4):
            P[j,j] = 1
        if i != index:
            P[i,i] = 0
            P[index,index] = 0
            P[index,i] = 1
            P[i,index] = 1
        Q = P@Q
        Qt = Q@P
        #print(P@Q)
        print(cluster_centers)
        print(index)
    print(Q)
    print(estimator.transition_matrix)
    print(Qt@matrix)
    assert_array_equal(A, estimator.transition_matrix)
    """


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

