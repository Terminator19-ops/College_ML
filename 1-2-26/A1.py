import numpy as np
from sklearn.datasets import load_iris


def dot_product(A, B):
    result = 0
    for a, b in zip(A, B):
        result += a * b
    return result

def euclidean_norm(A):
    sum_sq = 0
    for a in A:
        sum_sq += a * a
    return sum_sq ** 0.5

iris = load_iris()

X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

A = X[0]
B = X[1]

print("Own Dot:", dot_product(A, B))
print("NumPy Dot:", np.dot(A, B))

print("Own Norm:", euclidean_norm(A))
print("NumPy Norm:", np.linalg.norm(A))
