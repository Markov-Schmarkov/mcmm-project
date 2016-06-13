r"""
This module should handle the discretization by means of a kmeans or regspace clustering.
"""

#TODO exception handling

from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type
from random import sample
import numpy as np
from scipy.spatial import distance
from scipy.stats import rv_discrete

class KMeans(object):
    '''
    Class providing simple k-Means clustering for (n,d)-shaped 2-dimensional ndarray objects containing float data.
    '''

    def __init__(self,data,k,max_iter=100,method='forgy',metric='euclidean',atol=1e-05,rtol=1e-08):
        '''
        Args:
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
        self.method = method
        self.metric = metric
        self.rtol = rtol
        self.atol = atol
        self._cluster_centers = None
        self._cluster_labels = None
        self._cluster_dist = None

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

    @property
    def cluster_dist(self):
        if self._cluster_dist is None:
            self.fit()
        return self._cluster_dist
    @cluster_dist.setter
    def cluster_dist(self,value):
        self._cluster_dist = value

    def fit(self):
        '''
        Runs the clustering iteration on the data it was given when initialized.

        Cluster centers and cluster labels for the given data will be stored in the objects properties.
        '''

        cluster_centers = initialize_centers(self.data,self.k,self.method)

        counter = 0

        while counter < self.max_iter:
            cluster_labels, cluster_dist = get_cluster_info(self.data,cluster_centers,metric=self.metric)
            new_cluster_centers = set_new_cluster_centers(self.data,cluster_labels,self.k)
            if np.allclose(cluster_centers,new_cluster_centers,self.atol,self.rtol):
                print('terminated by break condition.')
                cluster_centers = new_cluster_centers
                break
            cluster_centers = new_cluster_centers
            counter = counter+1


        cluster_labels, cluster_dist = get_cluster_info(self.data,cluster_centers,metric=self.metric)
        print('%s iterations until termination.'%str(counter))
        self._cluster_centers = cluster_centers
        self._cluster_labels = cluster_labels
        self._cluster_dist = cluster_dist

    def transform(self,data):
        '''
        Returns cluster labeling for additional data corresponding
        to existing cluster centers stored in the object. (Also fits to initial data, if not fitted before)
        Args:
            data: (n,d)-shaped 2-dimensional ndarray
        Returns: cluster labels for passed data argument and cluster distances with respect to the given metric
        '''

        if self.cluster_centers is None or self.cluster_labels is None:
            self.fit()

        return get_cluster_info(data,self.cluster_centers,metric=self.metric)

    def fit_transform(self,add_data):
        '''
        Fits cluster centers based on given intial data PLUS additional data and returns centers, labeling and distances
        of the complete data set

        NOTE: this method does not change the stored properties self.cluster_centers and self.cluster_labels
        and is intended to make examination of changes of cluster information due to additional data possible.

        Args:
            add_data: additional (n,d)-shaped 2-dimensional ndarray containing float data.
            The second dimension d has to match the second dimension of the inital data.

        Returns:
            cluster centers, labeling and distances
        '''

        #TODO exception handling for dimension problems

        #if self.cluster_centers is None or self.cluster_labels is None:
        #   self.fit()

        data = np.vstack([self.data,add_data])

        cluster_centers = initialize_centers(data,self.k,self.method)

        counter = 0
        print('additional data fit:')
        while counter < self.max_iter:
            cluster_labels, cluster_dist = get_cluster_info(data, cluster_centers, metric=self.metric)
            new_cluster_centers = set_new_cluster_centers(data, cluster_labels, self.k)
            if np.allclose(cluster_centers, new_cluster_centers, self.atol, self.rtol):
                print('terminated by break condition.')
                cluster_centers = new_cluster_centers
                break
            cluster_centers = new_cluster_centers
            counter = counter + 1

        cluster_labels, cluster_dist = get_cluster_info(data, cluster_centers, metric=self.metric)
        print('%s iterations until termination.' % str(counter))
        return cluster_centers, cluster_labels, cluster_dist

#--------------
#global functions
#--------------


def get_cluster_info(data,cluster_centers,metric='euclidean'):
    '''
    For (n,d)-shaped float data and given centroids, returns the corresponding cluster centers and corresponding labeling.

    Args:
        data: (n,d) ndarray
        cluster_centers: (k,d) ndarray
        metric: metric parameters used as in scipy.spatial.distance.cdist. uses euclidean metric as default.

    Returns:
        cluster_labels: (d,1) vector containing the corresponding cluster centers of each of the data rows
        cluster_dist (d,1) vector containing squared distance of data observation to corresponding cluster centroid
    '''
    distance_matrix = distance.cdist(data,cluster_centers,metric)
    cluster_labels = np.argmin(distance_matrix,axis=1)
    cluster_dist = np.min(distance_matrix,axis=1)
    return cluster_labels, cluster_dist

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

def initialize_centers(data,k,method):
    '''
    initializes cluster centers with respect to given method
    '''

    if method == 'forgy':
        cluster_centers = forgy_centers(data,k)
    elif method == 'kmeans++':
        cluster_centers = kmeans_plusplus_centers(data,k)
    return cluster_centers

#---------
#cluster initializations
#---------

def forgy_centers(data,k):
    '''
    returns k randomly chosen cluster centers from data
    '''
    return sample(list(data),k)

def kmeans_plusplus_centers(data,k):
    '''
    returns cluster centers initialized by kmeans++ method,
    see http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
    '''

    index_vals = range(len(data)-1)
    center_list = []
    c1 = sample(list(data),1)
    if k == 1:
        return c1
    center_list.append(c1)
    while len(center_list)<k:
        distances = get_cluster_info(np.asmatrix(data),np.asmatrix(center_list))
        D2 = D2_weighting(distances)
        distribution = rv_discrete(values=(index_vals,D2))
        center_choice = data[distribution.rvs(size=1),:]
        center_list.append(center_choice)
    return np.array(center_list)

def D2_weighting(dist_array):
    '''
    performs the D^2-probability weighting on an ndarray of cluster distances associated to data points,
    see http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
    returns kmeans++ probability distribution vector
    '''
    D2 = dist_array**2
    sum = np.sum(D2)
    D2 = D2/sum
    return D2




