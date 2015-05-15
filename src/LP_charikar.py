import numpy as np
from gurobipy import *
import LoadData
from sklearn.metrics.pairwise import pairwise_distances

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
                vols[j] += y[i].X
                U[j].append(i)
                
    return y_assign, U, vols

# Greedily match clients based on distance
def match(C_sub, distances):
    unmatched = set(C_sub)
    matching = {}

    while len(unmatched) > 0:
        if len(unmatched) == 1:
            j = unmatched.pop()
            matching[j] = j
        else:
            # Find closest unmatched pair
            min_dist = float('inf')
            min_pair = None
            for i in unmatched:
                for j in unmatched:
                    if i != j:
                        if distances[i][j] < min_dist:
                            min_dist = distances[i][j]
                            min_pair = [i,j]

            # match that pair
            matching[min_pair[0]] = min_pair[1]
            matching[min_pair[1]] = min_pair[0]
            unmatched.remove(min_pair[0])
            unmatched.remove(min_pair[1])

    return matching

# sample a list of centers
def sample(y, y_assign, U, vols, matching, size):
    medoids = []
    n = len(y)
    sampled = [0 for _ in range(n)]
    while True:
        medoids = []
        for i in range(n):
            # If y not in a client subset, sample with LP value
            if y_assign[i] == n:
                if np.random.uniform(0,1) <= y[i].X:
                    medoids.append(i)
            else:
                j = y_assign[i]
                if not sampled[j]:
                    k = matching[j]

                    # If j not matched use vol of subset assigned to j
                    if j == k:
                        if np.random.uniform(0,1) <= vols[j]:
                            medoids.extend(U[j])
                    # Otherwise, use both vols of the matched pair
                    else:
                        uni = np.random.uniform(0,1)
                        if uni < 1-vols[k]:
                            medoids.extend(U[j])
                        elif uni < 2 - vols[k] - vols[j]:
                            medoids.extend(U[k])
                        else:
                            medoids.extend(U[k])
                            medoids.extend(U[j])
                    sampled[j] = 1
                    sampled[k] = 1
        if len(medoids) == size:
            break
    return np.array(medoids)             
                    
def convert_to_int(X, y, k, distances, random):
    print k
    C_sub = client_subset(X, y, distances)
    y_assign, U, vols = bundle(C_sub, X, y, distances)
    matching = match(C_sub, distances)
    while True:
        medoids = sample(y, y_assign, U, vols, matching, k)
        if len(medoids) == k:
            break
    return medoids 
    

def cluster_hier(distances, kvals, random=False):
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
        medoids =  convert_to_int(x, y, k, distances, random)
        assignment = assign_points_to_clusters(medoids, distances)
        cost = sum([distances[j][assignment[j]] for j in range(n)])
        all_costs[m-i-1] = cost
        all_centers[m-i-1] = medoids

    return all_centers, all_costs

        

if __name__ == '__main__':
    #centers = [[1, 1], [-1, -1], [1, -1]]
    X, y,n,k = LoadData.LoadData("C:\\Users\\ajp336\\Dropbox\\approx-clustering\\data\\Gaussian2\\Gauss_3_5_0.txt")
    distances = pairwise_distances(X)
    #distances = np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])
    print cluster_hier(distances,[4,3,2],1)
    

