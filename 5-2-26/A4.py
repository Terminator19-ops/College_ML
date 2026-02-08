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

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

x_vals = np.arange(0, 10.1, 0.1)
y_vals = np.arange(0, 10.1, 0.1)
X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
X_test = np.c_[X_mesh.ravel(), Y_mesh.ravel()]

y_pred = knn.predict(X_test)

plt.figure(figsize=(10, 8))
plt.scatter(X_test[y_pred == 0, 0], X_test[y_pred == 0, 1], c='lightblue', s=1, alpha=0.3)
plt.scatter(X_test[y_pred == 1, 0], X_test[y_pred == 1, 1], c='lightcoral', s=1, alpha=0.3)
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', s=200, edgecolors='black', zorder=5)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', s=200, edgecolors='black', zorder=5)
plt.xlabel('X Feature')
plt.ylabel('Y Feature')
plt.title('kNN Classification (k=3) Decision Boundary')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Blue: {np.sum(y_pred == 0)}, Red: {np.sum(y_pred == 1)}")
