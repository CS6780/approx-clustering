import numpy as np
from sklearn import metrics
import kmedoids
import multiswaps
import math

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances

# Calls k-median algorithm
def k_cluster_alg(k, distances, alg = ["k-medoids",None]):
    if alg[0] == "multiswaps":
        centers, cost = multiswaps.cluster(distances, k, alg[1])
    else:
        clusters, centers = kmedoids.cluster(distances, k)
        distances_to_centers = distances[:,centers]
        cost = sum( np.min(distances_to_centers, axis=1))
              
    return centers, cost

# Finds the solutions with costs in buckets [lb, ub] with the fewest
# cluster centers
def bucketed_solns(distances, beta, beta_0, alg):
    n = distances.shape[0]
    all_centers = [[] for i in range(n)]
    all_costs = [0 for i in range(n)]
    
    for k in range(1, n):
        k_centers, k_cost = k_cluster_alg(k, distances, alg)
        if k != 1 and k_cost > all_costs[k-1]:
            all_centers[k] = all_centers[k-1]
            all_costs[k] = all_costs[k-1]
        else:
            all_centers[k] = k_centers
            all_costs[k] = k_cost
    
    bucketed_centers = []
    lb = 0
    ub = beta_0
    k = n
    while k>1:
        next_bucket = [i for i in range(1,n) if all_costs[i]<= ub and \
                       all_costs[i] > lb]
        if len(next_bucket) > 0 and len(next_bucket) != n:
            k = min(next_bucket)
            bucketed_centers.append(all_centers[k])
        lb = ub
        ub = ub*beta

    return bucketed_centers

def replacement_center(centers, points, distances):
    i = np.argmin([ sum(distances[points, p]) for p in centers])
    return centers[i]
    

# Nests the new cluster centers with the old ones
def nest(curr_centers, curr_clusters, next_centers, distances, Z):
    n = len(curr_clusters)
    
    closest_centers_i = np.argmin(distances[next_centers,:][:,curr_centers], 1)
    closest_centers = [curr_centers[i] for i in closest_centers_i]
    new_centers = list(set(closest_centers))

    rem_centers = [x for x in curr_centers if x not in new_centers]
    new_clusters = curr_clusters
    k = max(curr_clusters)
    
    for y in rem_centers:
        k+=1
        # Find points to merge
        c1 = curr_clusters[y]
        points1 = [i for i in range(n) if curr_clusters[i] == c1]
        
        x = replacement_center(new_centers, points1, distances)
        c2 = curr_clusters[x]
        points2 = [i for i in range(n) if curr_clusters[i] == c2]

        # Relabel with k
        for p in points1:
            curr_clusters[p] = k
        for p in points2:
            curr_clusters[p] = k
        Z[k-n] = [c1, c2, x]
        
    return new_centers, new_clusters, Z
        
    
    
def cluster(distances, alg = ["k-medoids", None], random = False):
    n = distances.shape[0]
    if random:
        beta = 6.355
        beta_0 = beta^(np.random.uniform(0,1,1))
    else:
        beta = 3+math.sqrt(3)
        beta_0 = 1

    Z = np.zeros(shape=(n-1, 3))
    curr_centers = [i for i in range(n)]
    curr_clusters = [i for i in range(n)]
    k = n

    centers = bucketed_solns(distances, beta, beta_0, alg)
    
    for i in range(len(centers)):
        # Find next bucket's representative
        next_centers = centers[i]

        # Nest Solutions
        curr_centers, curr_clusters, Z = nest(curr_centers, curr_clusters, \
                                         next_centers, distances, Z)
        
    return Z
    
if __name__ == '__main__':
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=10, centers=centers, cluster_std=0.5,
                                random_state=999)
    distances = pairwise_distances(X)

    print cluster(distances, alg = ["multiswaps",1])
        

    
    
