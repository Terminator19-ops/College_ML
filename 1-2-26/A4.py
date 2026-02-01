import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def minkowski_distance(A, B, p):
    total = 0
    for a, b in zip(A, B):
        total += abs(a - b)**p
    return total**(1/p)

iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

A = X[0]
B = X[1]

ps = range(1, 11)
distances = []
for p in ps:
    d = minkowski_distance(A, B, p)
    distances.append(d)

plt.plot(ps, distances, marker='o')
plt.xlabel("p")
plt.ylabel("Distance")
plt.title("Minkowski Distance vs p")
plt.show()
