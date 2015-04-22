
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
# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=10, centers=centers, cluster_std=0.5,
#                             random_state=999)


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
          print("old ", psiS, " new ", psiNew, "difference ", psiNew - psiS)
          print("old fvalue ", f(Sc, X), "f value ", f(newSc,X))
          return newS
  return S

def centersToIndex(X, centers):
  centersDic = {}
  for c in centers:
    for i in range(len(X)):
      if c == X:
        centersDic[c] = i
        break
  return centersDic


def supermodular(X, k, epsilon=10,h_perc=100):
  S = initialS(X, k, len(X))
  Sc = X.difference(S)
  print("initial f ", f(Sc,X))
  while True:
    newS = updateS(S,Sc,X, epsilon,h_perc)
    if newS == S:
      print("Finished with f(S) = ", f(Sc, X))
      return Sc
    else:
      S = newS
      Sc = X.difference(S)

def supermodular_list(data, k, epsilon=10, h_perc=100):
  L = list(data)
  # print data
  # print L
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


# print supermodular(toFrozenSet(X))
# print supermodular_list(X,3,80)

# print "curvature is ", totalCurvature(toFrozenSet(X))




# print totalCurvature(toFrozenSet(X))
# print D
# print X
