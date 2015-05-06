import numpy as np
from sklearn import metrics
import kmedoids
import multiswaps
import math

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances

def next_assignment(Z, k, assignment):
    n= len(assignment)
    c1 = Z[n-k-1][0]
    c2 = Z[n-k-1][1]
    for j in range(n):
        if assignment[j] == c2:
            assignment[j] = int(c1)
    return assignment

# Calls k-median algorithm
def k_cluster_alg(k, distances, alg, warm_start = None):
    if alg[0] == "multiswaps":
        clusters, centers = multiswaps.cluster(distances, k, alg[1])
    elif alg[0] == "multiswaps-plus":
        clusters, centers = multiswaps.cluster2(distances, k, alg[1])
    else:
        clusters, centers = kmedoids.cluster(distances, k, 1000, warm_start)
    distances_to_centers = distances[:,centers]
    cost = sum( np.min(distances_to_centers, axis=1))
              
    return centers, cost

# Finds the solutions with costs in buckets [lb, ub] with the fewest
# cluster centers
def bucketed_solns(distances, beta, beta_0, alg, logn = False):
    n = distances.shape[0]
    all_centers = []
    all_costs = []

    k = 1
    while k < n:
        if k == 1:
            k_centers, k_cost = k_cluster_alg(k, distances, alg)
        else:
            warm_start = k_centers
            non_centers = [i for i in range(n) if i not in k_centers]
            warm_start = np.append(warm_start,np.random.choice(\
                non_centers,size=k-len(k_centers), replace=False))
            k_centers, k_cost = k_cluster_alg(k, distances, alg, warm_start)
            
            
        if k == 1 or k_cost < all_costs[-1]:
            all_centers.append(k_centers)
            all_costs.append(k_cost)
            
        if logn:
            k = 2*k
        else:
            k += 1
    
    bucketed_centers = []
    lb = 0
    ub = beta_0
    k = n
    while k>1:
        next_bucket = [i for i in range(len(all_costs)) if all_costs[i]<= ub and \
                       all_costs[i] > lb]
        if len(next_bucket) > 0 and len(next_bucket) != n:
            x = min(next_bucket)
            k = len(all_centers[x])
            bucketed_centers.append(all_centers[x])
        lb = ub
        ub = ub*beta
        
    return bucketed_centers

def replacement_center(centers, points, distances):
    i = np.argmin([ sum(distances[points, p]) for p in centers])
    return centers[i]
    

# Nests the new cluster centers with the old ones
def nest(curr_centers, curr_clusters, next_centers, distances, Z):
    n = distances.shape[0]
    k = len(curr_centers)
    
    closest_centers_i = np.argmin(distances[next_centers,:][:,curr_centers], 1)
    closest_centers = [curr_centers[i] for i in closest_centers_i]
    new_centers = list(set(closest_centers))

    rem_centers = [x for x in curr_centers if x not in new_centers]
    new_clusters = curr_clusters
    
    for y in rem_centers:
        # Find points to merge
        points1 = [i for i in range(n) if curr_clusters[i] == y]
        
        x = replacement_center(new_centers, points1, distances)
        # Relabel with x
        for p in points1:
            new_clusters[p] = x
            
        Z[n-k] = [x, y]
        k -= 1
        
    return new_centers, new_clusters, Z
        
    
    
def cluster(distances, alg = ["k-medoids", None], random = False, logn = False):
    n = distances.shape[0]
    if random:
        beta = 6.355
        beta_0 = beta**(np.random.uniform(0,1,1))
    else:
        beta = 3+math.sqrt(3)
        beta_0 = 1

    Z = np.zeros(shape=(n-1, 2))
    curr_centers = [i for i in range(n)]
    curr_clusters = [i for i in range(n)]
    k = n

    centers = bucketed_solns(distances, beta, beta_0, alg, logn)
    
    for i in range(len(centers)):
        # Find next bucket's representative
        next_centers = centers[i]

        # Nest Solutions
        curr_centers, curr_clusters, Z = nest(curr_centers, curr_clusters, \
                                         next_centers, distances, Z)
        
    return Z
    
if __name__ == '__main__':
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=50, centers=centers, cluster_std=0.5,
                                random_state=999)
    distances = pairwise_distances(X)

    Z= cluster(distances,["multiswaps",1],0,0)
    #print Z
    n=6
    assignment = [i for i in range(n)]
    for k in range(n-1,0,-1):
        assignment = next_assignment(Z,k,assignment)
        #print assignment
    
        

    
    
