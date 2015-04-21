import random
import itertools
import numpy as np

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

def predict(medoids, distances):
    n = distances.shape[0]
    return [ find_closest_median(i, medoids, distances) for i in range(n) ]

def cluster(distances, k, p, epsilon=0):
    n = distances.shape[0]
    medoids = random.sample(range(n), k)
    improvement = True
    while improvement:
        medoids, improvement = search_for_swap(medoids, distances, p, epsilon=epsilon)

    return predict(medoids, distances), medoids



