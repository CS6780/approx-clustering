import numpy as np
from gurobipy import *
import LoadData
from sklearn.metrics.pairwise import pairwise_distances

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

def convert_to_int(y_vals, k, random):
    n = len(y_vals)
    y_val_sort = np.argsort(y_vals)[::-1]
    if random:
        medoids = np.zeros(shape=(k))
        l = 0
        num_k = 0
        while num_k < k:
            if (np.random.uniform(0,1) < y_vals[y_val_sort[l]]) or \
               (n-l <= k-num_k):
                medoids[num_k] = y_val_sort[l]
                num_k += 1
            l += 1
        medoids = medoids.astype(int)
    else:
        medoids = y_val_sort[0:k]
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
        y_vals = [y[l].X for l in range(n)]
        medoids =  convert_to_int(y_vals, k, random)
        assignment = assign_points_to_clusters(medoids, distances)
        cost = sum([distances[j][assignment[j]] for j in range(n)])
        all_costs[m-i-1] = cost
        all_centers[m-i-1] = medoids

    return all_centers, all_costs

        

if __name__ == '__main__':
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, y,n,k = LoadData.LoadData("C:\\Users\\ajp336\\Dropbox\\approx-clustering\\data\\Gaussian2\\Gauss_3_5_0.txt")
    distances = pairwise_distances(X)
    print cluster_hier(distances,[7,6,5,4,3,2,1],1)

