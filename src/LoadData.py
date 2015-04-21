import scipy.sparse as sp
import numpy as np

# LoadData takes in a file path (string, a boolean cluster_loc
# representing whether the cluster variable is the first (1)
# or last entry (0) of each file line, and a boolean split
# representing whether the file is space delimited (1) or
# comma delimited (0)

def LoadData(filepath, cluster_loc = 1, split = 1):
    f = open(filepath)

    # First find number of samples and dimension
    n = sum([1 for line in f])
    f.seek(0)

    if split:
        dim = len(f.readline().split())-1
    else:
        dim = len(f.readline().split(','))-1
    f.seek(0)
        
    y = np.zeros(n)
    M = np.zeros((n,dim))
    clusterDict = dict()

    # Iterate through each sample
    i = 0
    k = 0    
    while True:
        if split:
            line = f.readline().split()
        else:
            line = f.readline().split(',')
            
        if len(line) < dim+1:
            break
        
        M[i] = [float(line[j+cluster_loc]) for j in range(dim)]
        cluster = line[cluster_loc-1]
        if cluster[-1:] == "\n":
            cluster = cluster[:-1]

        if cluster in clusterDict:
            y[i] = clusterDict[cluster]
        else:
            y[i] = k
            clusterDict[cluster] = k
            k += 1
            
        i += 1
        
    X =  sp.csr_matrix(M)
    X = X.toarray()
    return X, y, n, k
