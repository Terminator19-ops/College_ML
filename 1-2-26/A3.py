import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

feature = X[:, 0]

hist, bins = np.histogram(feature, bins=10)
plt.hist(feature, bins=10)
plt.title("Feature Density")
plt.show()

print("Mean:", np.mean(feature))
print("Variance:", np.var(feature))
