import random
import itertools

import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances

import matplotlib.pyplot as plt
from itertools import cycle


def find_closest_median(i, medoids, distances):
    return medoids[np.argmin(distances[i, medoids])]

def objective(medoids, distances):
    n = distances.shape[0]
    s = 0
    for i in range(0, n):
        s += distances[i, find_closest_median(i, medoids, distances)]
    return s

def swap(medoids, A, B):
    tmp = medoids
    for i, j in zip(A, B):
        tmp = swap_single(tmp, i, j)
    return tmp

def swap_single(medoids, a, b):
    return [b if i == a else i for i in medoids]

def search_for_swap(medoids, distances, p=2, epsilon=5):
    n = distances.shape[0]
    prev_objective = objective(medoids, distances)
    improvement = False
    for A in itertools.combinations(medoids, p):
        for B in itertools.combinations(range(n), p):
            new_medoids = swap(medoids, A, B)
            if objective(new_medoids, distances) < prev_objective - epsilon:
                improvement = True
                print("Improvement: %f" % (prev_objective - objective(new_medoids, distances)))
                return new_medoids, improvement
    return medoids, improvement

def cluster(distances, k, p):
    n = distances.shape[0]
    medoids = random.sample(range(n), k)
    improvement = True
    while improvement:
        medoids, improvement = search_for_swap(medoids, distances, p)

    return medoids, objective(medoids, distances)

if __name__ == '__main__':
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                                random_state=999)
    distances = pairwise_distances(X)
    n = distances.shape[0]
    
    #Choose random medoids
    # medoids = random.sample(range(n), num_clusters)
    num_clusters = 3
    p = 2
    medoids = [0, 1, 2]
    print(medoids, objective(medoids, distances))

    improvement = True
    while improvement:
        medoids, improvement = search_for_swap(medoids, distances, p)

    print(medoids, objective(medoids, distances))
