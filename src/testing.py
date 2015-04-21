import numpy as np

import multiswaps
import kmedoids
import submodular_demo
from LoadData import LoadData

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import adjusted_mutual_info_score

def objective(distances, assignment):
    n = distances.shape[0]
    return np.sum([ distances[i, assignment[i]] for i in range(n) ])

def test_clustering(cluster_alg, distances, labels_true, N=10):
    obj_results = []
    ami_results = []
    for _ in range(N):
        assignment, _ = cluster_alg()
        obj_results.append(objective(distances, assignment))
        ami_results.append(adjusted_mutual_info_score(y, assignment))

    avg_obj = np.mean(obj_results)
    avg_ami = np.mean(ami_results)
    min_obj = np.min(obj_results)
    min_ami = np.min(ami_results)
    max_obj = np.max(obj_results)
    max_ami = np.max(ami_results)

    return avg_obj, avg_ami, min_obj, min_ami, max_obj, max_ami

if __name__ == '__main__':
    X, y, n, k = LoadData("data/Real Data Sets/soybean-small.data.txt", cluster_loc=0, split=0)
    distances = pairwise_distances(X)

    p = 3
    multi_swaps = lambda: multiswaps.cluster(distances, k, p, epsilon=1)
    avg_obj, avg_ami, min_obj, min_ami, max_obj, max_ami = test_clustering(multi_swaps, distances, y, N=10)
    print("%s-swap Average: Objective: %f AMI: %f" % (p, avg_obj, avg_ami))
    print("%s-swap Best Case: Objective: %f AMI: %f" % (p, min_obj, max_ami))
    print("%s-swap Worst Case: Objective: %f AMI: %f" % (p, max_obj, min_ami))

    # k_medoids = lambda: kmedoids.cluster(distances, k=k)
    # avg_obj, avg_ami, min_obj, min_ami, max_obj, max_ami = test_clustering(k_medoids, distances, y, N=100)
    # print("K-Medoids Average: Objective: %f AMI: %f" % (avg_obj, avg_ami))
    # print("K-Medoids Best Case: Objective: %f AMI: %f" % (min_obj, max_ami))
    # print("K-Medoids Worst Case: Objective: %f AMI: %f" % (max_obj, min_ami))

    # super_modular = lambda: (submodular_demo.supermodular_list(distances), None)
    # avg_obj, avg_ami, min_obj, min_ami, max_obj, max_ami = test_clustering(super_modular, distances, y, N=1)
    # print("K-Medoids Average: Objective: %f AMI: %f" % (avg_obj, avg_ami))
    # print("K-Medoids Best Case: Objective: %f AMI: %f" % (min_obj, max_ami))
    # print("K-Medoids Worst Case: Objective: %f AMI: %f" % (max_obj, min_ami))














