from mcmm import clustering as cl
import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kmeans_blobs_2d(n_samples,n_clusters,k,method='kmeans++',std=1):
    '''
    generates random dataset by sklearn.datasets.samplesgenerator.make_blobs
    and visualizes the mcmm.analysis.KMeans clustering algorithm via pyplot

        Args:
        n_samples: number of observations in dataset
        n_clusters: number of clusters in dataset
        k: number of cluster centers to be determined by k-means
        method: the KMeans method, i.e. 'forgy' or 'kmeans++'
        std: the cluster intern standard deviation of the generated dataset
    '''

    data = make_blobs(n_samples,2,n_clusters,cluster_std=std)[0]
    kmeans = cl.KMeans(data,k,method)
    cluster_centers = kmeans.cluster_centers

    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='r', s=50)
    plt.show()


def kmeans_blobs_3d(n_samples,n_clusters,k,method='kmeans++',std=1):
    '''
    generates random dataset by sklearn.datasets.samplesgenerator.make_blobs
    and visualizes the mcmm.analysis.KMeans clustering algorithm via pyplot

        Args:
        n_samples: number of observations in dataset
        n_clusters: number of clusters in dataset
        k: number of cluster centers to be determined by k-means
        method: the KMeans method, i.e. 'forgy' or 'kmeans++'
        std: the cluster intern standard deviation of the generated dataset
    '''

    data = make_blobs(n_samples,3,n_clusters,cluster_std=std)[0]
    kmeans = cl.KMeans(data,k,method)
    cluster_centers = kmeans.cluster_centers

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0], data[:, 1],data[:,2])
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],cluster_centers[:,2], c='r', s=100,depthshade=False)
    plt.show()

kmeans_blobs_3d(300,3,3,method='kmeans++',std=1)