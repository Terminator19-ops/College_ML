import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def euclidean_norm(A):
    sum_sq = 0
    for a in A:
        sum_sq += a * a
    return sum_sq ** 0.5

def knn_predict(X_train, y_train, x_test, k):
    distances = []
    for x, y in zip(X_train, y_train):
        diff = x - x_test
        dist = euclidean_norm(diff)
        distances.append((dist, y))
    
    distances.sort(key=lambda x: x[0])
    labels = [y for _, y in distances[:k]]
    return max(set(labels), key=labels.count)

iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

own_preds = [knn_predict(X_train, y_train, x, 3) for x in X_test]
print("Custom KNN Predictions:", own_preds)
