r"""
This module should handle the discretization by means of a kmeans or regspace clustering.
"""

from __future__ import division
from random import sample
import numpy as np
from scipy.spatial import distance


class KMeans(object):
    '''
    class providing simple k-Means clustering for (n,d)-shaped 2-dimensional ndarray objects containing float data
    '''

    def __init__(self,data,k,max_iter=100,method='forgy',metric='euclidean'):
        '''
        input
        data: (n,d)-shaped 2-dimensional ndarray objects containing float data
        k: int, number of cluster centers. required to be <= n.
        max_iter: int, maximal iterations before terminating
        method: way of initializing cluster centers, default set to Forgy's method
        '''
        self.k = k
        self.max_iter = max_iter
        self.data = data
        self.method=method
        self.metric=metric
        self._cluster_centers = None
        self._cluster_labels = None

    @property
    def cluster_centres(self):
        if cluster_centres is None:
            cluster_centers = self.fit()[0]
            cluster_labels = self.fit()[1]
        return self._cluster_centers

    @property
    def cluster_assignment(self):
        if cluster_assigntment is None:
            cluster_centers = self.fit()[0]
            cluster_labels = self.fit()[1]
        return self._cluster_assignment

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

    def wcss(cluster_labels,cluster_dist):
        '''
        returns within-cluster-sum-of-squares in a (k,1) shaped vector for each cluster
        '''
        wcss_vec = []
        for i in range(self.k):
            ss = cluster_dist[cluster_labels == i]
            wcss_vec.append(sum(ss**2))
        return wcss_vec

    def forgy_centers(data):
        '''
        returns k randomly chosen cluster centers from data
        '''
        return sample(data,self.k)

    def optimize_centroid(cluster_points):
        '''
        for a given set of observations in one cluster, compute and return a new centroid
        '''
        vecsum = np.sum(cluster_points,axis=0)
        centroid = vecsum/cluster_points.shape[0]
        return centroid

    def fit():
        '''
        '''
        #TODO actual clustering loop



