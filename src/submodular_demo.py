
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances

import numpy as np
import math
import random

# np.random.seed(17)

# H_PERC = 100



##############################################################################



##############################################################################
# Define Functions

# Find closest median in S to i
def find_closest_median(i, S, distances):
    closest = None
    closest_dist = float("inf")
    
    for j in S:
        if distances[i][j] < closest_dist:
            closest_dist = distances[i][j]
            closest = j
        
    return closest

# Returns the objective function for S
def f(S, distances, F):
    if F is not None:
        if S in F:
            return F[S]
    
    n = distances.shape[0]
    if S == frozenset():
        Sum = max([f(frozenset([j]),distances,F) for j in range(n)])
    else:
        Sum = 0 
        for x in range(n):
            if x not in S:
                distx = [distances[x][s] for s in S]
                Sum+=min(distx)
    if F is not None:      
        F[S] = Sum
    return Sum

# Curvature Calculation

def curvatureValue(X, j, fX, fEmpty, distances, FDict = None):
    n = distances.shape[0]
    Xj = X.difference([j])
    
    num = (fX - f(Xj,distances,FDict))
    denom = (f(frozenset([j]),distances,FDict)-fEmpty)
    
    if denom == 0: 
        return float("inf")
    else:
        return num/float(denom)


def totalCurvature(distances, F = None):
    n = distances.shape[0]
    X = frozenset([i for i in range(n)])
  
    fX = f(X,distances,F)
    fEmpty = f(frozenset(),distances,F)
    vals = [curvatureValue(X,j, fX, fEmpty, distances,F) for j in X]
    return 1- min(vals)

# The linear part of the submodular function
def l(A, distances, F):
    fEmpty = f(frozenset(), distances, F)
    vals = [f(frozenset([j]), distances, F) for j in A]
    return sum(vals)-fEmpty*len(A)

# The non-linear part of the submodular function
def g(A, distances, F):
    n = distances.shape[0]
    ADiff = frozenset([i for i in range(n) if i not in A])
    return -l(A, distances, F) - f(ADiff, distances, F)

# Draw a random sample for a subset of A
def drawB(A):
    uniformRandom = np.random.random_sample()
    p = math.log(uniformRandom*(math.e-1)+1)
    B = []
    for x in A:
        if np.random.random_sample()<p:
            B.append(x)
    return frozenset(B)

# Estimate part of the potential function representing g(S)
def estimateH(A, distances, H, F, h_perc):
    if A in H:
        return H[A]
    Sum = 0
    for x in range(h_perc):
        Sum+= g(drawB(A),distances, F)
        
    H[A] = Sum/float(h_perc)
    return Sum/float(h_perc)

# potential function
def psi(A, distances, H, F, h_perc):
    return (1-1.0/math.e)*estimateH(A, distances, H, F, h_perc) + \
           l(A,distances,F)

# updateS looks for a local move that increases phi
def updateS(S, psiOld, distances, sorted_distances, H, F, delta, h_perc):
    n = distances.shape[0]
    k = len(S)

    SList = list(S)
    ScList = [i for i in range(n) if i not in S]
    Sc = frozenset(ScList)
    ScDist = np.zeros(shape=(n,k))

    # Look at localized swaps
    for j in range(min(50,k/10)): 
        for i in Sc:
            newi = sorted_distances[i][j]
            if newi not in S:
                newS = S.difference([ScDist[i][j]]).union([i])
                psiNew = psi(newS, distances, H, F, h_perc)
                if psiNew>= psiOld+delta:
                    # print "old ", psiOld, " new ", psiNew, "difference ", psiNew - psiOld
                    return newS, psiNew, True
            
    # Sample random swaps
    iters = 0
    while iters < min(100,n*n):
        iters += 1
        i = np.random.choice(SList)
        j = np.random.choice(ScList)
        newS = S.difference([i]).union([j])
        psiNew = psi(newS, distances, H, F, h_perc)
        if psiNew>= psiOld+delta:
            # print "old ", psiOld, " new ", psiNew, "difference ", psiNew - psiOld
            return newS, psiNew, True
        
    return list(S), psiOld, False

# find a random initial set represented as a frozenset
def initialS(k,n):
    S = np.random.choice([i for i in range(n)], size=k, replace=False)
    return frozenset(S)


# supermodular looks for a locally optimal solution with a potential function
def cluster(distances, k, warm_start= None, epsilon=5, h_perc=50):
    # Find Initial S
    n = distances.shape[0]
    X = frozenset([i for i in range(n)])
    if warm_start is None:
        S = initialS(n-k,n)
    else:
        S = X.difference(frozenset(warm_start))

    # Calculate Delta
    HDict = {}
    FDict = {}
    vg = max([g([j],distances,FDict) for j in range(n)])
    vl = max([abs(l([j],distances,FDict)) for j in range(n)])
    delta = max(vg, vl)*epsilon/float(n)

    sorted_distances = np.zeros(shape=(n,n))
    for j in range(n):
        sorted_distances[j] = [i[0] for i in sorted(enumerate(distances[j]),\
                                                    key=lambda x:x[1])]

    # Local search
    psiS = psi(S,distances,HDict,FDict,h_perc)
    bestS = S
    bestF = f(X.difference(S),distances,FDict)
    while True:
        newS, newPsi, updated = updateS(S, psiS, distances, sorted_distances,\
                                        HDict, FDict, delta, h_perc)
        if not updated:
            break
        else:
            # Update S and best seen subset
            S = newS
            psiS = newPsi
            newf = f(X.difference(S),distances,FDict)
            if newf < bestF:
                bestF = newf
                bestS = S
                
    Sc = X.difference(bestS)
    # print "Finished with f(S) = ", bestF
    clusters = [find_closest_median(i,Sc,distances) for i in range(n)]
    curvature = totalCurvature(distances, FDict)

    return clusters, Sc

if __name__ == '__main__':
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=100, centers=centers, cluster_std=0.5,
                            random_state=999)
    distances = pairwise_distances(X)
    
    print(cluster(distances,3,None,10,100,100))

    print("curvature is ", totalCurvature(distances))
