from mcmm import clustering as cl
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import unittest
from nose.tools import assert_true, assert_false, assert_equals, assert_raises
from numpy.testing import assert_array_equal


def test_find_cluster_centers():
    """This test should check if the cluster_centers are reasonable
    We test clustering in R^1
    """
    n = 200 #number of data points
    k =15 #number of cluster_centers
    factor = 0.25 #how much is the data perturbed
    data = np.zeros((n,1))
    for i in range(0,n):
        data[i] = i % k + factor * np.random.rand() * math.pow(-1,int(2*np.random.rand()))
    #fig,ax = plt.subplots(ncols=2,nrows=1)
    plt.scatter(data[:,0],np.zeros((n,1)))
    plt.scatter(data[:,0],np.ones((n,1)))
    plt.scatter(data[:,0],2*np.ones((n,1)))
    
    clustering = cl.KMeans(data,k)
    clustering2 = cl.KMeans(data,k,method='kmeans++')
    clustering3 = cl.KMeans(data,k,method='kmeans++')
    
    cluster_centers = clustering.cluster_centers
    cluster_labels = clustering.cluster_labels
    cluster_centers2 = clustering2.cluster_centers
    cluster_labels2 = clustering2.cluster_labels
    cluster_centers3 = clustering3.cluster_centers
    cluster_labels3 = clustering3.cluster_labels
    plt.scatter(cluster_centers[:],np.zeros((k,1)),c='r')
    plt.scatter(cluster_centers2[:],np.ones((k,1)),c='r')
    plt.scatter(cluster_centers3[:],2*np.ones((k,1)),c='r')
    #check if clusterlabels are reasonable
    for j in range(0,k): 
        index = np.argmin(cluster_centers)
        zahl = int(cluster_centers[index])
        if cluster_centers[index] - zahl > 0.5:
            zahl = zahl +1
        cluster_centers[index] = cluster_centers[index] + k+1
        #assert_equals(zahl, j)