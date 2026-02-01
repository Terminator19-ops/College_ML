import numpy as np
import matplotlib.pyplot as plt
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

def accuracy(y_true, y_pred):
    correct = sum(y_true == y_pred)
    return correct / len(y_true)

iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

ks = range(1, 12)
accs = []

for k in ks:
    preds = [knn_predict(X_train, y_train, x, k) for x in X_test]
    acc = accuracy(y_test, np.array(preds))
    accs.append(acc)

plt.plot(ks, accs, marker='o')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs k")
plt.show()
