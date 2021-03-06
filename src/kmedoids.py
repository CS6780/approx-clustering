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

def cluster_hier(distances, kvals, max_iter=1000, warm_start=None, epsilon=0.1):
    m = len(kvals)
    n = distances.shape[0]
    all_costs = [0 for i in range(m)]
    all_centers = [[0 for j in range(kvals[i])] for i in range(m)]

    for i in range(m):
        k = kvals[i]
        if k == 1:
            k_clusters, k_centers = cluster(distances, k, max_iter)
            k_cost = sum([distances[j][k_clusters[j]] for j in range(n)])
        else:
            warm_start = k_centers
            non_centers = [l for l in range(n) if l not in k_centers]
            warm_start = np.append(warm_start,np.random.choice(\
                non_centers,size=k-len(k_centers), replace=False))
            k_clusters, k_centers = cluster(distances, k, max_iter, warm_start)
            k_cost = sum([distances[j][k_clusters[j]] for j in range(n)])
            
        all_costs[i] = k_cost
        all_centers[i] = k_centers
        
    return all_centers, all_costs





