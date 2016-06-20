from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
from random import sample
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
        [n_samples,dim] = self.data.shape
        visited = np.zeros(n_samples,dtype=bool)
        cluster_labels = [None]*n_samples
        cluster_index = 0

        for i,observation in enumerate(self.data):
            if visited[i]:
                pass
            else:
                visited[i] = True
                p = self._data[i]
                neighbors = get_region(self._data,p,self._eps,self._metric)
                if neighbors.shape < self._minPts:
                    # mark as noise
                    cluster_labels[i] = 0
                else:
                    # move up to next cluster
                    cluster_index = cluster_index + 1
                    expand_cluster()

        self.cluster_labels = cluster_labels



#------------
#global functions
#------------

def get_region(data,p,eps,metric):
    '''
    returns subset of data containing all points in the eps-ball around p with respect to given metric
    '''
    n_samples,dim  = data.shape
    distances = distance.cdist(p.reshape(1,dim),data,metric=metric).reshape(dim)
    mask = distances<eps
    indices = np.arange(n_samples)[mask]
    region = data[mask]

    return region

def expand_cluster(data,i,neighbor_indices,neighbors,cluster_labels,cluster_index,eps,minPts,visited,metric):
    '''

    '''
    cluster_labels[i] = cluster_index
    for k,p in enumerate(neighbors):
        visited[k] = True
        neighbors_indices2,neighbors2 = get_region(data,p,eps,metric)
        if neighbors2.shape[0] >= minPts:
            new_indices = np.union1d(neighbor_indices,neighbors_indices2)
        if visited[k] is None:
            cluster_labels[k] = cluster_index










