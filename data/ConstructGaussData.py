import numpy as np
import numpy.random as rand
import math
import os

def ConstructGaussian(dim, k, noisy, n):
    directory = os.getcwd()
    filename = "\\Gaussian2\\Gauss_"+str(dim)+"_"+str(k)+"_"+str(noisy)+".txt"
    f = open(directory+filename, 'w+')
    
    identity = np.zeros((dim,dim))
    for i in range(dim):
        identity[i][i] = 1
    
    total = 0
    for i in range(k):
        center = rand.uniform(-1,1,dim)
        size = int(math.floor(rand.uniform(0,1,1)*(2*n/float(k))))
        size = max(1, size)
        total+=size

        for j in range(size):
            stdev = 0.05
            if 100*rand.uniform(0,1) < noisy:
                stdev = 0.25
                
            x = rand.multivariate_normal(center, stdev*identity)
            f.write(str(i)+" ")
            for l in range(dim):
                f.write(str(x[l])+" ")
            f.write("\n")

    f.close()

    
        
        
        
    
