import numpy as np
from sklearn import metrics
import kmedoids
import LP_greedy
import multiswaps
import math
import LoadData

#from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances

def next_assignment(Z, k, assignment):
    n= len(assignment)
    c1 = Z[n-k-1][0]
    c2 = Z[n-k-1][1]
    for j in range(n):
        if assignment[j] == c2:
            assignment[j] = int(c1)
    return assignment


# Finds the solutions with costs in buckets [lb, ub] with the fewest
# cluster centers
def bucketed_solns(distances, beta, beta_0, alg, logn = False):
    n = distances.shape[0]
    if logn:
        p = int(math.floor(math.log(n,2)))
        kvals = [2**i for i in range(p+1)]
    else:
        kvals = [i for i in range(1,n)]

    if alg[0] == "multiswaps-plus":
        all_centers, all_costs = multiswaps.cluster_hier(distances, kvals, alg[1])
    elif alg[0] == "k-medoids":
        all_centers, all_costs = kmedoids.cluster_hier(distances, kvals, 1000)
    elif alg[0] == "LP-greedy-rand":
        kvals.reverse()
        all_centers, all_costs = LP_greedy.cluster_hier(distances, kvals, 1)
    else:
        kvals.reverse()
        all_centers, all_costs = LP_greedy.cluster_hier(distances, kvals, 0)
            

    min_cost = (1.0/np.min([all_costs[i] for i in range(len(all_costs)) if all_costs[i] != 0]))
    all_costs = [min_cost*all_costs[i] for i in range(len(all_costs))]
    
    bucketed_centers = []
    lb = 0
    ub = beta_0
    k = n
    while k>1:
        if k == n and np.min(all_costs) == 0:
            next_bucket = [i for i in range(len(all_costs)) if all_costs[i] == 0]           
        else:
            next_bucket = [i for i in range(len(all_costs)) if all_costs[i]< ub and \
                           all_costs[i] >= lb]
            
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
    m = len(rem_centers)
    rep_centers = [0 for _ in range(m)]
    rep_costs = [0 for _ in range(m)]
    new_clusters = curr_clusters
    
    for i in range(m):
        y = rem_centers[i]
        # Find points to merge and rep center/cost
        points = [j for j in range(n) if curr_clusters[j] == y]
        x = replacement_center(new_centers, points, distances)
        rep_centers[i] = x
        rep_costs[i] = sum([distances[j][x]-distances[j][y] for j in range(n)])

        # relabel
        for p in points:
            new_clusters[p] = x
            
    # smartly merge
    by_cost = np.argsort(rep_costs)
    for i in range(m):
        y = rem_centers[by_cost[i]]
        x = rep_centers[by_cost[i]]
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
    X, y,n,k = LoadData.LoadData("C:\\Users\\ajp336\\Dropbox\\approx-clustering\\data\\Gaussian2\\Gauss_3_5_0.txt")
    distances = pairwise_distances(X)

    Z= cluster(distances,["LP-greedy",None],0,0)
    
    assignment = [i for i in range(n)]
    for k in range(n-1,145,-1):
        assignment = next_assignment(Z,k,assignment)
        cost = sum([distances[i][assignment[i]] for i in range(n)])
        

    
    
