import random
import itertools
import numpy as np
from scipy.special import binom
#import LoadpMedian
import kmedoids

#from sklearn.datasets.samples_generator import make_blobs
#from sklearn.metrics.pairwise import pairwise_distances

def find_closest_median(i, medoids, distances):
    closest = None
    closest_dist = float("inf")
    
    for j in medoids:
        if distances[i][j] < closest_dist:
            closest_dist = distances[i][j]
            closest = j
        
    return closest

def objective(medoids, distances, FDict):
    n = distances.shape[0]

    if medoids in FDict:
        return FDict[medoids]
        
    Sum = 0
    for x in range(n):
        if x not in medoids:
            distx = [distances[x][s] for s in medoids]
            Sum+=min(distx)
    FDict[medoids] = Sum
    return Sum

def swap(medoids, A, B):
    tmp = medoids
    for i, j in zip(A, B):
        tmp = swap_single(tmp, i, j)
    return tmp

def swap_single(medoids, a, b):
    return [b if i == a else i for i in medoids]

def search_for_swap(medoids, obj, distances, FDict, p=2, delta=0):
    n = distances.shape[0]
    prev_objective = obj
    medoids_list = list(medoids)
    medoids_c_list = [i for i in range(n) if i not in medoids]
    medoids_c = frozenset(medoids_c_list)
    
    improvement = False
    for q in range(1,p+1):
        if q > len(medoids):
            break
        #    for A in itertools.combinations(medoids, q):
        #        for B in itertools.combinations(medoids_c, q):
        #            new_medoids = medoids.difference(A).union(B)
        #            new_objective = objective(new_medoids, distances, FDict)
        #            if new_objective < (1.0-delta)*prev_objective:
        #                improvement = True
        #                #print("Improvement: %f" % (prev_objective - new_objective))
        #                return new_medoids, new_objective, improvement
        #else:
        #print q
        iters = 0
        while iters < n*q:
            iters += 1
            A = np.random.choice(medoids_list,size=q,replace=False)
            B = np.random.choice(medoids_c_list,size=q,replace=False)
            new_medoids = medoids.difference(A).union(B)
            new_objective = objective(new_medoids, distances, FDict)
            if new_objective < (1.0-delta)*prev_objective:
                improvement = True
                #print("Improvement: %f" % (prev_objective - new_objective))
                return new_medoids, new_objective, improvement
                
    return medoids, prev_objective, improvement

def predict(medoids, distances):
    n = distances.shape[0]
    return [ find_closest_median(i, medoids, distances) for i in range(n) ]

def cluster(distances, k, p, warm_start = None, epsilon=0):
    n = distances.shape[0]
    p = min(k,p)
    delta = epsilon/float(binom(n-k,p)*binom(k,p))
    #print delta
    if warm_start is None:
        medoids = frozenset(np.random.choice([i for i in range(n)],size=k,replace=False))
    else:
        medoids = frozenset(warm_start)
        
    FDict = {}
    obj = objective(medoids, distances, FDict)
    
    improvement = True
    while improvement:
        medoids, obj, improvement = search_for_swap(medoids, obj, distances, FDict, \
                                                    p, delta)

    return predict(medoids, distances), list(medoids)

def cluster2(distances, k, p, warm_start = None, epsilon=0):
    n = distances.shape[0]
    p = min(k,p)
    delta = epsilon/float(binom(n-k,p)*binom(k,p))
    #print delta
    if warm_start is None:
        medoids = frozenset(np.random.choice([i for i in range(n)],size=k,replace=False))
    else:
        medoids = frozenset(warm_start)
        
    FDict = {}
    obj = objective(medoids, distances, FDict)
    
    improvement = True
    while improvement:
        medoids, obj, improvement = search_for_swap(medoids, obj, distances, FDict, \
                                                    p, delta)
    
    return kmedoids.cluster(distances, k, max_iter=10, warm_start = list(medoids))

##if __name__ == '__main__':
##    centers = [[1, 1], [-1, -1], [1, -1]]
##    X, labels_true = make_blobs(n_samples=100, centers=centers, cluster_std=0.5,
##                            random_state=999)
##    distances = pairwise_distances(X)
##
##    # distances, n, k = LoadpMedian.LoadpMedian("C:\\Users\\ajp336\\Dropbox\\approx-clustering\\data\\p-median Instances\\pmed22.txt")
##    cluster2(distances,3,1,None,0.1)



