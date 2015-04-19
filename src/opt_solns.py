import numpy as np
import os
from LoadpMedian import *
from LoadData import *
from gurobipy import *
from sklearn.metrics.pairwise import pairwise_distances

def kmedian_opt(distances, IP, k1, k2):
    model = Model("k-median")
    n = np.shape(distances)[0]
    y,x = {}, {}

    if IP:
        var_type = GRB.BINARY
    else:
        var_type = GRB.CONTINUOUS
        
    for j in range(n):
        y[j] = model.addVar(obj=0, vtype=var_type, name="y[%s]"%j)
        for i in range(n):
            x[i,j] = model.addVar(obj=distances[i,j], vtype=var_type, \
                                  name="x[%s,%s]"%(i,j))
                
    model.update()
    
    for i in range(n):
        coef = [1 for j in range(n)]
        var = [x[i,j] for j in range(n)]
        model.addConstr(LinExpr(coef,var), "=", 1, name="Assign[%s]"%i)
        
    for j in range(n):
        for i in range(n):
            model.addConstr(x[i,j], "<", y[j], name="Strong[%s,%s]"%(i,j))
            
    coef = [1 for j in range(n)]
    var = [y[j] for j in range(n)]

    model.setParam( 'OutputFlag', False )
    model.__data = x,y
    outputs = [0 for i in range(n+1)]
    model.update()
    
    for k in range(k1,k2+1):
        k_constr = model.addConstr(LinExpr(coef,var), ">=", rhs=k, name="k_median")
        model.update()
        model.optimize()

        if model.status == GRB.status.OPTIMAL:
            outputs[k] =  model.objVal
        else:
            outpus[k] = 0
            
    return outputs

def write_opt_bounds(distances, filepath, IP=1):
    f = open(filepath, 'w+')
    n = np.shape(distances)[0]

    bounds = kmedian_opt(distances, IP)
    for k in range(1,n+1):
        f.write(str(k)+" "+str(bound[k])+"\n")

def write_opt_pmedians(path_files, file_bounds):
    g = open(file_bounds, 'w+')
    
    for f in os.listdir(path_files):
        print f
        distances, n, k = LoadpMedian(path_files+"\\"+f)
        bound = kmedian_opt(distances, 1, k, k)
        g.write(f+" "+str(bound)+"\n")

def write_opt_data(path_files, file_bounds, name):
    g = open(file_bounds, 'w+')
    
    for f in os.listdir(path_files):
        print f
        X, n, k = LoadData(path_files+"\\"+f, name)
        distances = pairwise_distances(X)
        bound = kmedian_opt(distances, 1, k, k)
        g.write(f+" "+str(bound)+"\n")
        
    
    
