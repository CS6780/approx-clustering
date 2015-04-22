
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import numpy as np
import math
import random

# np.random.seed(17)

# H_PERC = 100



##############################################################################



##############################################################################
# Define Functions
def toFrozenSet(X):
  XList = X.tolist()
  XListOfTuples = [tuple(l) for l in XList]
  XFrozenSet = frozenset(XListOfTuples)
  return XFrozenSet

def d(x,y):
  return np.linalg.norm(np.array(x)-np.array(y))

D = {}

def f(S,X):
  if S in D:
    return D[S]
  if S == frozenset():
    values = [f(frozenset([a]), X) for a in X]
    D[S]= min(values)
    return min(values)
  Sum = 0
  for x in X:
    if x not in S:
      distances = [d(x, s) for s in S]
      Sum+=min(distances)
  D[S]=Sum
  return Sum

def without(S, x):
  #returns a copy of S without x without changing S
  return S.difference([x])

def curvatureValue(X, j, fX, fEmpty):
  num = (fX - f(without(X,j),X))
  denom = (f(frozenset([j]),X)-fEmpty)
  if denom == 0: 
    return float("inf")
  else:
    return num/denom


def totalCurvature(X):
  #expects X to be a frozenset
  fX = f(X,X)
  fEmpty = f(frozenset(),X)
  vals = [curvatureValue(X,j, fX, fEmpty) for j in X]
  return 1- min(vals)

def l(A, X):
  fEmpty = f(frozenset(),X)
  vals = [f(frozenset([j]), X) for j in A]
  return sum(vals)-fEmpty

def g(A,X):
  return -l(A,X) - f(X.difference(A),X)

def drawP():
  uniformRandom = np.random.random_sample()
  return math.log(uniformRandom*(math.e-1)+1)

def drawB(A):
  p = drawP()
  B = set()
  for x in A:
    if np.random.random_sample()<p:
      B.add(x)
  return frozenset(B)

HDict ={}
def estimateH(A,X, h_perc):
  if A in HDict:
    return HDict[A]
  Sum = 0
  for x in range(h_perc):
    Sum+= g(drawB(A),X)
  HDict[A] = Sum/h_perc
  return Sum/h_perc

def psi(A,X, h_perc):
  return (1-1/math.e)*estimateH(A,X,h_perc) + l(A,X)

def initialS(X, k, N):
  i = N-k
  S = set()
  for x in X:
    if i ==0:
      break
    S.add(x)
    i-=1
  return frozenset(S)

def updateS(S,Sc, X, epsilon,h_perc):
  delta = epsilon/len(X)
  psiS = psi(S, X,h_perc)
  for a in S:
      for b in Sc:
        newS = S.difference([a]).union([b])
        newSc = Sc.difference([b]).union([a])
        psiNew = psi(newS,X,h_perc)
        if psiNew>= psiS+delta:
          print "old ", psiS, " new ", psiNew, "difference ", psiNew - psiS
          print "old fvalue ", f(Sc, X), "f value ", f(newSc,X)
          return newS
  return S

def centersToIndex(X, centers):
  centersDic = {}
  for c in centers:
    for i in range(len(X)):
      if c == tuple(X[i]):
        centersDic[c] = i
        break
  return centersDic


def supermodular(X, k, epsilon=10,h_perc=100):
  S = initialS(X, k, len(X))
  Sc = X.difference(S)
  print "initial f ", f(Sc,X)
  while True:
    newS = updateS(S,Sc,X, epsilon,h_perc)
    if newS == S:
      print "Finished with f(S) = ", f(Sc, X)
      return Sc
    else:
      S = newS
      Sc = X.difference(S)

def supermodular_list(data, k, epsilon=10, h_perc=100):
  L = list(data)
  centers = supermodular(toFrozenSet(data), k, epsilon, h_perc)
  centersDic = centersToIndex(L, centers)
  closestCenter =[]
  for x in L:
    currentCenter = 0
    currentDistance = float("inf")
    for y in centers:
      if(d(x,y)<currentDistance):
        currentCenter=y
        currentDistance=d(x,y)
    closestCenter+=[centersDic[currentCenter]]
  return closestCenter

# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=10, centers=centers, cluster_std=0.5,
#                             random_state=999)
# # print supermodular(toFrozenSet(X))
# print supermodular_list(X,3,80)

# print "curvature is ", totalCurvature(toFrozenSet(X))




# print totalCurvature(toFrozenSet(X))
# print D
# print X


##############################################################################
# Compute Affinity Propagation
# af = AffinityPropagation(preference=-50).fit(X)
# cluster_centers_indices = af.cluster_centers_indices_
# labels = af.labels_

# n_clusters_ = len(cluster_centers_indices)

# print('Estimated number of clusters: %d' % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

# ##############################################################################
# # Plot result
# import matplotlib.pyplot as plt
# from itertools import cycle

# plt.close('all')
# plt.figure(1)
# plt.clf()

# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     class_members = labels == k
#     cluster_center = X[cluster_centers_indices[k]]
#     plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
#     for x in X[class_members]:
#         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.savefig("affinity-propogation-demo.png")
# plt.clf()
