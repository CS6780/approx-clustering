import random
import itertools

import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import adjusted_mutual_info_score

import matplotlib.pyplot as plt
from itertools import cycle

from LoadData import LoadData


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

def score(labels_true, labels_pred):
    return adjusted_mutual_info_score(labels_true, labels_pred)

def search_for_swap(medoids, distances, p=2, epsilon=0):
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

def predict(medoids, distances):
    n = distances.shape[0]
    return [ np.argmin(distances[i, medoids]) for i in range(n) ]

if __name__ == '__main__':
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # n = 100
    # X, y = make_blobs(n_samples=n, centers=centers, cluster_std=0.5,
    #                             random_state=999)
    # k = 3


    # X, y, n, k = LoadData("data/Gaussian/Gauss_10_5_0.txt")
    X, y, n, k = LoadData("data/Real Data Sets/iris.data.txt", cluster_loc=0, split=0)
    # print(X, y, n, k)

    distances = pairwise_distances(X)
    # n = distances.shape[0]
    
    #Choose random medoids
    medoids = random.sample(range(n), k)
    p = 2
    print(medoids, objective(medoids, distances))

    improvement = True
    while improvement:
        medoids, improvement = search_for_swap(medoids, distances, p)

    print(medoids, objective(medoids, distances))
    labels_pred = predict(medoids, distances)
    print(y, labels_pred)
    print(score(y, labels_pred))



