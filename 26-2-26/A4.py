import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data[iris.target != 2] # type:ignore
y = iris.target[iris.target != 2] # type:ignore

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X_train)

print("Labels:", kmeans.labels_)
print("Cluster Centers:")
print(kmeans.cluster_centers_)
print("Inertia:", kmeans.inertia_)
