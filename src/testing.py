import numpy as np
import timeit
import multiswaps
import kmedoids
import submodular_demo
from LoadData import LoadData
from LoadpMedian import LoadpMedian

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import adjusted_mutual_info_score

def objective(distances, assignment):
    n = distances.shape[0]
    return np.sum([ distances[i, assignment[i]] for i in range(n) ])

def test_clustering(cluster_alg, distances, labels_true, N=10):
    time_results = []
    obj_results = []
    ami_results = []
    for _ in range(N):
        start_time = timeit.default_timer()
        assignment, _ = cluster_alg()
        time_results.append(timeit.default_timer()-start_time)
        obj_results.append(objective(distances, assignment))
        ami_results.append(adjusted_mutual_info_score(labels_true, assignment))

    avg_tim = np.mean(time_results)
    avg_obj = np.mean(obj_results)
    avg_ami = np.mean(ami_results)
    min_tim = np.min(time_results)
    min_obj = np.min(obj_results)
    min_ami = np.min(ami_results)
    max_tim = np.max(time_results)
    max_obj = np.max(obj_results)
    max_ami = np.max(ami_results)

    return avg_tim, avg_obj, avg_ami, min_tim, min_obj, min_ami, \
           max_tim, max_obj, max_ami

# if __name__ == '__main__':
    
    # X, y, n, k = LoadData("C:\\Users\\ajp336\\Dropbox\\approx-clustering\\data\\Real Data Sets\\soybean-small.data.txt", cluster_loc=0, split=0)
    # distances = pairwise_distances(X)

    # p = 3
    # multi_swaps = lambda: multiswaps.cluster(distances, k, p, epsilon=1)
    # avg_tim, avg_obj, avg_ami, min_tim, min_obj, min_ami, max_tim, \
    #          max_obj, max_ami = test_clustering(multi_swaps, distances, y, N=10)
    # print("%s-swap Average: Time: %f Objective: %f AMI: %f" % (p, avg_tim, avg_obj, avg_ami))
    # print("%s-swap Best Case: Time: %f Objective: %f AMI: %f" % (p, min_tim, min_obj, max_ami))
    # print("%s-swap Worst Case: Time: %f Objective: %f AMI: %f" % (p, max_tim, max_obj, min_ami))

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














