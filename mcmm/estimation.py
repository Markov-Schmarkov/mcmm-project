r"""
This module should handle the transition counting and transition matrix estimation.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

import numpy as np

def count_matrix_from_cluster_labels_using_sliding_window(cluster_labels, lag_time, window_shift, num_cluster):
    """Computes the count matrix based on the cluster labels. The method is the sliding window approach.
    
    Parameters:
    cluster_labels: 1-dimensional numpy.ndarray, every entry gives the cluster number of the element
    lag_time: number, size of the window
    window_shift: number of trajectory indices by which windows is shifted	
    num_cluster: number of clusters
    """
    num_states = cluster_labels.shape[0]
    count_matrix = np.zeros((num_cluster,num_cluster))
    for s in range (0, num_states-lag_time, window_shift):
        count_matrix[cluster_labels[s],cluster_labels[s + lag_time]] +=1    
    return count_matrix	

def transition_matrix_from_count_matrix(count_matrix):
    row_sums = count_matrix.sum(axis=1)
    count_matrix = count_matrix / row_sums[:, np.newaxis]
    return count_matrix


def reversible_transition_matrix_from_count_matrix(count_matrix):
    matrix = transition_matrix_from_count_matrix(count_matrix)
    matrix_next = np.zeros(matrix.shape)
    while not np.allclose(matrix, matrix_next, rtol=0.01):
        for i,j in np.ndindex(*count_matrix.shape):
            d_i = count_matrix[i,:].sum() / matrix[i,:].sum()
            d_j = count_matrix[j,:].sum() / matrix[j,:].sum()
            matrix_next[i,j] = (count_matrix[i,j] + count_matrix[j,i])/(d_i + d_j)
        matrix, matrix_next = matrix_next, matrix
    return matrix_next
