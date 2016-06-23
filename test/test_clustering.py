from mcmm import clustering as cl
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import unittest
from nose.tools import assert_true, assert_false, assert_equals, assert_raises
from numpy.testing import assert_array_equal
from mcmm import example as ex


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
        
def test_find_cluster_centers_R2():
    """This test should check if the cluster_centers are reasonable
    We test clustering in R^2
    """
    n = 50 #number of data points in x- and y-direction
    k = 5 #number of cluster_centers in x- and y-direction when generating data
    k2 = k #number of cluster_centers in x- and y-direction when doing clustering
    #dim = 2
    factor = 0.3 #how much is the data perturbed
    data = np.zeros((n*n,2))
    for i in range(0,n):
        for j in range(0,n):
            data[n * i + j,0] = i % k + factor * np.random.normal() * math.pow(-1,int(2*np.random.rand()))
            data[n * i + j,1] = j % k + factor * np.random.normal() * math.pow(-1,int(2*np.random.rand()))
    #fig,ax = plt.subplots(ncols=2,nrows=1)
    plt.scatter(data[:,0],data[:,1])
    
    #Do more clustering and take the best clustering
    anzahl = k*4
    error = n*n*n
    for t in range(0,anzahl):
        errorNew = 0
        clusteringNew = cl.KMeans(data,k2*k2)
        cluster_centersNew = clusteringNew.cluster_centers
        for i in range(0,n):
            for j in range(0,n):
                dist = n
                for l in range(0,k2*k2):
                    distNew = np.linalg.norm(data[n*i+j,:] - cluster_centersNew[l,:])
                    if dist > distNew:
                        dist = distNew
                errorNew = errorNew + dist
        print(errorNew)
        print(error)
        if errorNew < error:
            error = errorNew
            clustering = clusteringNew
            cluster_centers = cluster_centersNew
    
    plt.scatter(cluster_centers[:,0],cluster_centers[:,1],c='r')
    
    #check if cluster_centers are reasonable
    for i in range(0,k):
        for j in range(0,k):
            check = 0
            for l in range(0,k2*k2):
                if np.linalg.norm([i,j] - cluster_centers[l,:]) < 0.2:
                    check = 1
            assert_equals(check,1)

def test_find_cluster_centers_example1():
    """This test uses the build in example class to test the clustering
    In the end we check if the error (sum of distance of all points to nearest cluster center)
    is smaller than expected
    """
    n = 5000 #number of data points
    k = 15 #number of cluster_centers
    factor = 0.25 #how much is the data perturbed
    x = ex.generate_test_data(n,1)
    data = x[0]
    #fig,ax = plt.subplots(ncols=2,nrows=1)
    plt.scatter(data[:,0],data[:,1])
    
    clustering = cl.KMeans(data,k,method='kmeans++')
    cluster_centers = clustering.cluster_centers
    cluster_labels = clustering.cluster_labels
    
    errorNew = 0
    for i in range(0,n):
        dist = n
        for l in range(0,k):
            distNew = np.linalg.norm(data[i,:] - cluster_centers[l,:])
            if dist > distNew:
                dist = distNew
        errorNew = errorNew + dist
    print(errorNew/n)
    print(4/k)
    check = 0
    if errorNew/n < 4/k:
        check = 1
    assert_equals(check,1)

    plt.scatter(cluster_centers[:,0],cluster_centers[:,1],c='r')