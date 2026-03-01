import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data[iris.target != 2] # type:ignore
y = iris.target[iris.target != 2] # type:ignore

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

distortions = []

for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_train)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(9, 5))
plt.plot(range(2, 20), distortions, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Distortion)")
plt.title("Elbow Plot for K-Means Clustering")
plt.xticks(range(2, 20))
plt.grid(True)
plt.show()

diffs = np.diff(distortions)
diffs2 = np.diff(diffs)
optimal_k = np.argmax(diffs2) + 3

print("Distortions:", distortions)
print("Optimal k (elbow):", optimal_k)
