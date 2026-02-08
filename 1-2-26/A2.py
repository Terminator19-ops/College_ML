import numpy as np
from sklearn.datasets import load_iris

def mean(data):
    total = sum(data)
    return total / len(data)

def variance(data):
    m = mean(data)
    sq_diff = [(x - m)**2 for x in data]
    return sum(sq_diff) / len(data)

def std_dev(data):
    var = variance(data)
    return var**0.5

iris = load_iris()
X = iris.data[iris.target != 2] # type: ignore
y = iris.target[iris.target != 2] # type: ignore

class0 = X[y == 0]
class1 = X[y == 1]

centroid0 = class0.mean(axis=0)
centroid1 = class1.mean(axis=0)

spread0 = class0.std(axis=0)
spread1 = class1.std(axis=0)

interclass_distance = np.linalg.norm(centroid0 - centroid1)

print("Interclass Distance:", interclass_distance)
