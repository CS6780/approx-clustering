#!/usr/bin/env python

import numpy as np
import random

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import adjusted_mutual_info_score

from LoadData import LoadData

def cluster(distances, k=3):
    m = distances.shape[0] # number of points

    # Pick k random medoids.
    curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])

    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)
    clusters = assign_points_to_clusters(curr_medoids, distances)
   
    # Until the medoids stop updating, do the following:
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)

        # Update cluster medoids to be lowest cost point. 
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]

    return clusters, curr_medoids

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster,cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)

if __name__ == '__main__':
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
    #                             random_state=999)
    # X, y, n, k = LoadData("data/Gaussian/Gauss_10_5_0.txt")
    X, y, n, k = LoadData("data/Real Data Sets/iris.data.txt", cluster_loc=0, split=0)
    distances = pairwise_distances(X)

    clusters, curr_medoids = cluster(distances, k=k)
    print(clusters, curr_medoids)
    print(adjusted_mutual_info_score(y, clusters))





