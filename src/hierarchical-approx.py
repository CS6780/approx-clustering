import numpy as np
from sklearn import metrics
import kmedoids
import math

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances

# Calls k-median algorithm
def k_cluster_alg(k, distances, alg = "k-medoids"):
    if alg == "k-medoids":
        clusters, centers = kmedoids.cluster(distances, k)
        distances_to_centers = distances[:,centers]
        cost = sum( np.min(distances_to_centers, axis=1))
              
    return clusters, centers, cost

# Finds the solution with cost in bucket [lb, ub] with the fewest
# cluster centers
def find_next_solution(k, lb, ub, distances, alg):
    clusters = []
    centers = []
    
    while True:
        k -= 1
        n_clusters, n_centers, n_cost = k_cluster_alg(k, distances, alg)
        if n_cost > ub:
            break
        else:
            clusters = n_clusters
            centers = n_centers
            
    return centers, clusters

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
        Z[k-n] = [c1, c2]
        
    return new_centers, new_clusters, Z
        
    
    
def cluster(distances, alg = "k-medoids", random = False):
    n = np.shape(distances)[0]
    if random:
        beta = 6.355
        beta_0 = beta^(np.random.uniform(0,1,1))
    else:
        beta = 3+math.sqrt(3)
        beta_0 = 1

    Z = np.zeros(shape=(n-1, 2))
    curr_centers = [i for i in range(n)]
    curr_clusters = [i for i in range(n)]
    cost_lb = 0
    cost_ub = beta_0
    k = n
    
    while k > 1:
        print curr_centers
        # Find next bucket's representative
        while True:     
            next_centers, next_clusters = find_next_solution(k,cost_lb, cost_ub, \
                                                             distances, alg)
            if len(next_centers) == 0:
                break
            else:
                cost_lb = cost_ub
                cost_ub = cost_ub*beta

        # Nest Solutions
        curr_centers, curr_clusters, Z = nest(curr_centers, curr_clusters, \
                                         next_centers, distances, Z)
        
        k = len(curr_centers)
        cost_lb = cost_ub
        cost_ub = cost_lb*beta
        
    return Z
    
if __name__ == '__main__':
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=10, centers=centers, cluster_std=0.5,
                                random_state=999)
    distances = pairwise_distances(X)

    print cluster(distances)
        

    
    
