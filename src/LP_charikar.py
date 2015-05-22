import numpy as np
from gurobipy import *
import LoadData
from sklearn.metrics.pairwise import pairwise_distances
import timeit

from itertools import izip
argmin = lambda array: min(izip(array, xrange(len(array))))[1]

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

# Find a subset of clients to consider
def client_subset(x, y, distances):
    n = len(y)

    # Calculate ave distances
    d_av = [0 for i in range(n)]
    for i in range(n):
        for j in range(n):
            if x[i,j].X > 0:
                d_av[i] += (y[j].X)*distances[i][j]

    C = [1 for _ in range(n)]
    d_av_sort = np.argsort(d_av)
    curr_min = 0
    C_sub = []

    while True:
        # Add j with min d_av to C_sub and remove from C
        j = d_av_sort[curr_min]
        C_sub.append(j)
        C[j] = 0

        # Remove from C any k s.t. d(j,k) < 4*d_av[k]
        for l in range(curr_min+1, n):
            k = d_av_sort[l]
            if distances[j][k] <= 4*d_av[k]:
                C[k] = 0

        # Update min
        for l in range(curr_min+1, n):
            if C[d_av_sort[l]] == 1:
                curr_min = l
                break
        if d_av_sort[curr_min] == j:
            break
        
    return C_sub

# bundles facilities to clients in C_sub
def bundle(C_sub, x, y, distances):
    n = distances.shape[0]
    y_assign = [int(n) for _ in range(n)]
    U = [[] for _ in range(n)]
    vols = [0 for _ in range(n)]

    for i in C_sub:
        # Find closest point to i in C_sub
        if len(C_sub) == 1:
            Ri = float('inf')
        else:
            Ri = 0.5*min([distances[i][k] for k in C_sub if k != i])

        # Assign all facilities serving i within 1.5*Ri to i
        for j in range(n):
            if (x[i,j].X > 0) and (distances[i][j] <= 1.5*Ri) and \
               (y_assign[j] == n):
                y_assign[j] = i
                vols[i] += y[j].X
                U[i].append(j)

    return y_assign, U, vols

# Greedily match clients based on distance
def match(C_sub, distances):
    unmatched = set(C_sub)
    matching = {}
    m = len(C_sub)
    sort_dists = [ np.argsort([distances[C_sub[i]][C_sub[j]] for j in range(i+1, m)]) \
                   for i in range(m)]
    min_index = [0 for _ in range(m-1)]
    mins = [distances[C_sub[i]][C_sub[sort_dists[i][0]+i+1]] for i in range(m-1)]
    mins.append(float('inf'))
    
    while len(unmatched) > 0:
        if len(unmatched) == 1:
            j = unmatched.pop()
            matching[j] = j
        else:
            # Find closest unmatched pair
            j = argmin(mins)
            k = sort_dists[j][min_index[j]]+j+1
            x = C_sub[j]
            y = C_sub[k]

            # match that pair
            matching[x] = y
            matching[y] = x
            unmatched.remove(x)
            unmatched.remove(y)
            mins[j] = float('inf')
            mins[k] = float('inf')

            # Update mins
            for i in range(m-1):
                if (C_sub[i] in unmatched):
                    while True:
                        if min_index[i] >= m-i-1:
                            mins[i] = float('inf')
                            break
                        elif C_sub[sort_dists[i][min_index[i]]+i+1] not in unmatched:
                            min_index[i] += 1
                        else:
                            break
    return matching

def open_bundle(U, y, vol):
    probs = [(y[l].X)/float(vol) for l in U]
    return np.random.choice(U, p=probs)

# sample a list of centers
def sample(y, y_assign, U, vols, matching, size):
    medoids = []
    n = len(y)

    while True:
        medoids = []
        sampled = [0 for _ in range(n)]
        for i in range(n):
            # If y not in a client subset, sample with LP value
            if y_assign[i] == n:
                if np.random.uniform(0,1) <= y[i].X:
                    medoids.append(i)
            else:
                j = y_assign[i]
                if sampled[j] == 0:
                    k = matching[j]
                    # If j not matched use vol of subset assigned to j
                    if j == k:
                        if np.random.uniform(0,1) <= vols[j]:
                            medoids.append(open_bundle(U[j],y, vols[j]))
                    # Otherwise, use both vols of the matched pair
                    else:
                        uni = np.random.uniform(0,1)
                        if uni < 1-vols[k]:
                            medoids.append(open_bundle(U[j],y, vols[j]))
                        elif uni < 2 - vols[k] - vols[j]:
                            medoids.append(open_bundle(U[k],y, vols[k]))
                        else:
                            medoids.append(open_bundle(U[j],y, vols[j]))
                            medoids.append(open_bundle(U[k],y, vols[k]))
                    sampled[j] = 1
                    sampled[k] = 1
                    
            
        if len(medoids) == size:
            break
        
    return np.array(medoids)             
                    
def convert_to_int(X, y, k, distances, max_iter):
    k = int(sum([y[i].X for i in range(distances.shape[0])]))
    C_sub = client_subset(X, y, distances)
    n = distances.shape[0]
    y_assign, U, vols = bundle(C_sub, X, y, distances)
    matching = match(C_sub, distances)

    best_med = []
    best_cost = float('inf')
    for i in range(max_iter):
        medoids = sample(y, y_assign, U, vols, matching, k)
        assignment = assign_points_to_clusters(medoids, distances)
        cost = sum([distances[j][assignment[j]] for j in range(n)])
        if cost < best_cost:
            best_cost = cost
            best_med = medoids
    return medoids, cost
    

def cluster_hier(distances, kvals, max_iter = 1000):
    model = Model("k-median")
    n = np.shape(distances)[0]
    y,x = {}, {}

    # Add variables
    var_type = GRB.CONTINUOUS
    for j in range(n):
        y[j] = model.addVar(obj=0, vtype=var_type, name="y[%s]"%j)
        for i in range(n):
            x[i,j] = model.addVar(obj=distances[i,j], vtype=var_type, \
                                  name="x[%s,%s]"%(i,j))
    model.update()

    # Add assignment constraints
    for i in range(n):
        coef = [1 for j in range(n)]
        var = [x[i,j] for j in range(n)]
        model.addConstr(LinExpr(coef,var), "=", 1, name="Assign[%s]"%i)

    # Add feasibility constraints   
    for j in range(n):
        for i in range(n):
            model.addConstr(x[i,j], "<", y[j], name="Strong[%s,%s]"%(i,j))
    model.setParam( 'OutputFlag', False )
    model.update()

    m = len(kvals)
    all_costs = [0 for i in range(m)]
    all_centers = [[0 for j in range(kvals[i])] for i in range(m)]
    for i in range(m):
        k = kvals[i]
        # Update k constraint & solve
        if i == 0:
            coef = [1 for j in range(n)]
            var = [y[j] for j in range(n)]
            k_constr = model.addConstr(LinExpr(coef,var), "<", rhs=k)
        else:
            k_constr.RHS = k
        model.update()
        model.optimize()

        # Round and get assignment/cost
        medoids, cost =  convert_to_int(x, y, k, distances, max_iter)
        all_costs[m-i-1] = cost
        all_centers[m-i-1] = medoids

    return all_centers, all_costs

        

if __name__ == '__main__':
    #centers = [[1, 1], [-1, -1], [1, -1]]
    X, y,n,k = LoadData.LoadData("C:\\Users\\ajp336\\Dropbox\\approx-clustering\\data\\Real Data Sets\\iris.data.txt", 0, 0)
    distances = pairwise_distances(X)
    #distances = np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])
    start_time = timeit.default_timer()
    kvals = [n-i for i in range(n)]
    cluster_hier(distances,kvals,1)
    print timeit.default_timer()-start_time

##    start_time = timeit.default_timer()
##    kvals = [i+1 for i in range(536, n)]
##    cluster_hier(distances,kvals,1)
##    print timeit.default_timer()-start_time

    #start_time = timeit.default_timer()
    #cluster_hier(distances,[12, 11, 10, 9, 8],1000)
    #print timeit.default_timer()-start_time
    

