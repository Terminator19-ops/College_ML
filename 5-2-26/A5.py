import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

try:
    data = np.load('training_data.npz')
    X_train, y_train = data['X'], data['y']
except FileNotFoundError:
    np.random.seed(42)
    X_feature = np.random.uniform(1, 10, 20)
    Y_feature = np.random.uniform(1, 10, 20)
    X_train = np.column_stack([X_feature, Y_feature])
    y_train = np.where(X_feature + Y_feature > 11, 1, 0)

x_vals = np.arange(0, 10.1, 0.1)
y_vals = np.arange(0, 10.1, 0.1)
X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
X_test = np.c_[X_mesh.ravel(), Y_mesh.ravel()]

k_values = [1, 3, 5, 7, 9, 15]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    ax = axes[idx]
    ax.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], c='lightblue', s=1, alpha=0.3)
    ax.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], c='lightcoral', s=1, alpha=0.3)
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', s=100, edgecolors='black', zorder=5)
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', s=100, edgecolors='black', zorder=5)
    ax.set_title(f'k = {k}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Effect of k on Decision Boundary')
plt.tight_layout()
plt.show()
