r"""
This module should handle the discretization by means of a kmeans or regspace clustering.
"""

#TODO kmeans++ cluster inititalization in addition to forgys method
#TODO visualization api?

from __future__ import division
from random import sample
import numpy as np
from scipy.spatial import distance


class KMeans(object):
    '''
    class providing simple k-Means clustering for (n,d)-shaped 2-dimensional ndarray objects containing float data
    '''

    def __init__(self,data,k,max_iter=100,method='forgy',metric='euclidean',atol=1e-05,rtol=1e-08):
        '''
        input
        data: (n,d)-shaped 2-dimensional ndarray objects containing float data
        k: int, number of cluster centers. required to be <= n.
        max_iter: int, maximal iterations before terminating
        method: way of initializing cluster centers, default set to Forgy's method
        metric: metric used to compute distances. for possible arguments see metric arguments of scipy.spatial.distance.cdist
        atol,rtol: absolute and relative tolerance threshold to stop iteration before reaching max_iter. see numpy.allclose documentation.
        '''
        self.k = k
        self.max_iter = max_iter
        self.data = data
        self.method = 'forgy'
        self.metric = 'euclidean'
        self.rtol = rtol
        self.atol = atol
        self._cluster_centers = None
        self._cluster_labels = None

    @property
    def cluster_centers(self):
        if self._cluster_centers is None:
            self.fit()
        return self._cluster_centers
    @cluster_centers.setter
    def cluster_centers(self,value):
        self._cluster_centers = value

    @property
    def cluster_labels(self):
        if self._cluster_labels is None:
            self.fit()
        return self._cluster_labels
    @cluster_labels.setter
    def cluster_labels(self,value):
        self._cluster_labels = value

    def fit(self):
        if self.method == 'forgy':
            cluster_centers = forgy_centers(self.data,self.k)

        counter = 0

        while counter < self.max_iter:
            cluster_labels, cluster_dist = get_cluster_info(self.data,cluster_centers,metric=self.metric)
            new_cluster_centers = set_new_cluster_centers(self.data,cluster_labels,self.k)
            if np.allclose(cluster_centers,new_cluster_centers,self.atol,self.rtol):
                print 'terminated by break condition.'
                cluster_centers = new_cluster_centers
                break
            cluster_centers = new_cluster_centers
            counter = counter+1


        print '%s iterations until termination.'%str(counter)
        self._cluster_centers = cluster_centers
        self._cluster_labels = cluster_labels


#--------------
#global functions
#--------------


def get_cluster_info(data,cluster_centers,metric='euclidean'):
    '''
    for (n,d)-shaped float data and given centroids, returns the corresponding cluster centers

    input
    data: (n,d) ndarray
    cluster_centers: (k,d) ndarray
    metric: metric parameters used as in scipy.spatial.distance.cdist. uses euclidean metric as default.

    output
    cluster_labels: (d,1) vector containing the corresponding cluster centers of each of the data rows
    cluster_dist (d,1) vector containing squared distance of data observation to corresponding cluster centroid
    '''
    distance_matrix = distance.cdist(data,cluster_centers,metric)
    cluster_labels = np.argmin(distance_matrix,axis=1)
    cluster_dist = np.min(distance_matrix)
    return cluster_labels, cluster_dist

def forgy_centers(data,k):
    '''
    returns k randomly chosen cluster centers from data
    '''
    return sample(data,k)

def optimize_centroid(cluster_points):
    '''
    for a given set of observations in one cluster, compute and return a new centroid
    '''
    vecsum = np.sum(cluster_points,axis=0)
    centroid = vecsum/cluster_points.shape[0]
    return centroid

def set_new_cluster_centers(data,cluster_labels,k):
    '''
    for given data and clusterlabeling, construct new centers for each cluster
    '''
    center_list = []
    for i in range(k):
        new_center = optimize_centroid(data[cluster_labels == i,:])
        center_list.append(new_center)

    return np.vstack(center_list)




