import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

iris = load_iris()
X = iris.data[iris.target != 2] # type:ignore
y = iris.target[iris.target != 2] # type:ignore

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X_train)

sil = silhouette_score(X_train, kmeans.labels_)
ch = calinski_harabasz_score(X_train, kmeans.labels_)
db = davies_bouldin_score(X_train, kmeans.labels_)

print("Silhouette Score :", sil)
print("CH Score         :", ch)
print("DB Index         :", db)
