import numpy as np
from gurobipy import *

    model = Model("k-median")
    n = np.shape(distances)[0]
    y,x = {}, {}
    
    for j in range(n):
        y[j] = model.addVar(obj=0, vtype="B", name="y[%s]"%j)
        for i in range(n):
    model.update()
    
    for i in range(n):
        model.addConstr(LinExpr(coef,var), "=", 1, name="Assign[%s]"%i)
        
    for j in range(n):
        for i in range(n):
            model.addConstr(x[i,j], "<", y[j], name="Strong[%s,%s]"%(i,j))
            
    model.addConstr(LinExpr(coef,var), "=", rhs=k, name="k_median")
    
    model.update()
    model.__data = x,y
    model.optimize()

    if model.status == GRB.status.OPTIMAL:
    else:
        return 0

