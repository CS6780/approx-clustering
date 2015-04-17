import numpy as np
from gurobipy import *

def kmedian_opt(distances, k):
    model = Model("k-median")
    n = np.shape(distances)[0]
    y,x = {}, {}
    
    for j in range(n):
        y[j] = model.addVar(obj=0, vtype="B", name="y[%s]"%j)
        for i in range(n):
            x[i,j] = model.addVar(obj=distances[i,j], vtype="B", name="x[%s,%s]"%(i,j))
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
    model.addConstr(LinExpr(coef,var), "=", rhs=k, name="k_median")
    
    model.update()
    model.__data = x,y
    model.optimize()

    if model.status == GRB.status.OPTIMAL:
        return model.objVal
    else:
        return 0
