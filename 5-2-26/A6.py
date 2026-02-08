import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data[iris.target != 2][:, :2]
y = iris.target[iris.target != 2]

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
X_mesh, Y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
X_test = np.c_[X_mesh.ravel(), Y_mesh.ravel()]

k_values = [1, 3, 5, 7, 9, 15]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X_test)
    acc = knn.score(X, y)
    
    ax = axes[idx]
    ax.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], c='lightblue', s=1, alpha=0.3)
    ax.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], c='lightcoral', s=1, alpha=0.3)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=100, edgecolors='black', zorder=5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=100, edgecolors='black', zorder=5)
    ax.set_title(f'k = {k}, Acc: {acc:.3f}')
    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.grid(True, alpha=0.3)

plt.suptitle('Iris Dataset: Effect of k on Decision Boundary')
plt.tight_layout()
plt.show()
