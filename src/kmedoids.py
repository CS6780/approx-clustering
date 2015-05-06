#!/usr/bin/env python

import numpy as np
import random

def cluster(distances, k=3, max_iter=1000, warm_start = None):
    m = distances.shape[0] # number of points
    # Pick k random medoids.
    if warm_start is None:
        curr_medoids = np.random.choice([i for i in range(m)],size=k,replace=False)
    else:
        curr_medoids = np.array(warm_start)
        
    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)
    clusters = assign_points_to_clusters(curr_medoids, distances)
    i = 0
   
    # Until the medoids stop updating, do the following:
    while not np.array_equal(old_medoids,curr_medoids) and i < max_iter:
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)

        # Update cluster medoids to be lowest cost point. 
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
        i += 1

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





