import numpy as np
import networkx as nx

def LoadpMedian(filepath):
    f = open(filepath)
    dims = f.readline().split()
    n = int(dims[0])
    m = int(dims[1])
    k = int(dims[2])

    G = nx.Graph()
    G.add_nodes_from([i+1 for i in range(n)])

    for j in range(m):
        line = f.readline().split()
        u = int(line[0])
        v = int(line[1])
        w = int(line[2])
        G.add_edge(u, v, weight = w)

    DistDict = nx.floyd_warshall(G)
    Distances = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            d = DistDict[i+1][j+1]
            Distances[i][j] = d
            Distances[j][i] = d

    return Distances, n, k
    

        
