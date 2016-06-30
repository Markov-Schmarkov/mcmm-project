from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
import numpy as np
from scipy.spatial import distance

class DBSCAN(object):

    def __init__(self,data,eps,minPts,metric='euclidean'):
        '''

        Args:
            data:
            eps:
            minPts:

        Returns:

        '''

        self._data = data
        self._eps = eps
        self._minPts = minPts
        self._cluster_labels = None
        self._metric = metric

    @property
    def cluster_labels(self):
        if self._cluster_labels is None:
                self.fit()
        return self._cluster_labels
    @cluster_labels.setter
    def cluster_labels(self,value):
        self._cluster_labels = value

    def fit(self):
        # initialize variables
        [n_samples,dim] = self._data.shape
        visited = np.zeros(n_samples,dtype=bool)
        cluster_labels = [None]*n_samples
        cluster_index = 0

        for i,observation in enumerate(self._data):
            if visited[i]:
                pass
            else:
                visited[i] = True
                neighbor_indices = get_region(self._data,observation,self._eps,self._metric)
                neighbors = self._data[neighbor_indices]
                if len(neighbor_indices) < self._minPts:
                    # mark as noise
                    cluster_labels[i] = 'noise'
                else:
                    # move up to next cluster
                    cluster_index = cluster_index + 1
                    #-------------
                    #expand cluster subalgorithm
                    #-------------
                    cluster_labels=expand_cluster(self._data,i,neighbor_indices,neighbors,cluster_labels,cluster_index,self._eps,self._minPts,visited,self._metric)

        self.cluster_labels = cluster_labels



#------------
#global functions
#------------

def get_region(data,p,eps,metric):
    '''
    returns subset of data containing all points in the eps-ball around p with respect to given metric and the
    corresponding indices
    '''
    n_samples,dim  = data.shape
    distances = distance.cdist(p.reshape(1,dim),data,metric=metric)
    mask = distances<eps
    mask = mask.reshape((n_samples,))
    indices = np.arange(n_samples)[mask]
    #region = data[mask]

    return indices

def expand_cluster(data,i,neighbor_indices,neighbors,cluster_labels,active_cluster_index,eps,minPts,visited,metric):
    '''

    '''
    cluster_labels[i] = active_cluster_index
    while not np.all(visited[neighbor_indices]):

        for k,p in enumerate(neighbors):

            if not visited[k]:
                visited[k] = True
                neighbor_indices2 = get_region(data,p,eps,metric)
                if len(neighbor_indices2) >= minPts:
                    neighbor_indices = np.union1d(neighbor_indices,neighbor_indices2)
            if cluster_labels[k] is None:
                cluster_labels[k] = active_cluster_index

    return cluster_labels









