import numpy as np
import matplotlib.pyplot as plt
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

k_values = range(2, 11)
sil_scores = []
ch_scores = []
db_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_train)
    labels = kmeans.labels_
    sil_scores.append(silhouette_score(X_train, labels))
    ch_scores.append(calinski_harabasz_score(X_train, labels))
    db_scores.append(davies_bouldin_score(X_train, labels))

k_list = list(k_values)

print("k  | Silhouette | CH Score   | DB Index")
for i, k in enumerate(k_list):
    print(f"{k:<3}| {sil_scores[i]:<11.4f}| {ch_scores[i]:<11.4f}| {db_scores[i]:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(k_list, sil_scores, marker='o')
axes[0].set_xlabel("k")
axes[0].set_ylabel("Silhouette Score")
axes[0].set_title("Silhouette Score vs k")

axes[1].plot(k_list, ch_scores, marker='o')
axes[1].set_xlabel("k")
axes[1].set_ylabel("CH Score")
axes[1].set_title("Calinski-Harabasz Score vs k")

axes[2].plot(k_list, db_scores, marker='o')
axes[2].set_xlabel("k")
axes[2].set_ylabel("DB Index")
axes[2].set_title("Davies-Bouldin Index vs k")

plt.tight_layout()
plt.show()
