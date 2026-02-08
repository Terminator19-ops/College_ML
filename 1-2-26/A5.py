import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial.distance import minkowski


def minkowski_distance(A, B, p):
    total = 0
    for a, b in zip(A, B):
        total += abs(a - b)**p
    return total**(1/p)

iris = load_iris()
X = iris.data[iris.target != 2] # type: ignore
y = iris.target[iris.target != 2] # type: ignore

A = X[0]
B = X[1]

for p in [1, 2, 3, 5]:
    own_dist = minkowski_distance(A, B, p)
    scipy_dist = minkowski(A, B, p)

    print(f"p = {p}")
    print("Own Minkowski Distance   :", own_dist)
    print("SciPy Minkowski Distance :", scipy_dist)
    print("-" * 40)
