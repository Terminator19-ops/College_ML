import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

X_features = X
y_labels = y

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y_labels,
    test_size=0.3,
    random_state=42,
    stratify=y_labels
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])
