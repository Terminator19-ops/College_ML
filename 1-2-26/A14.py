import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
X = iris.data[iris.target != 2] # type: ignore
y = iris.target[iris.target != 2] # type: ignore

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_aug = np.c_[np.ones(X_train.shape[0]), X_train]

W = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y_train

def linear_predict(x):
    x_with_bias = np.r_[1, x]
    pred = np.dot(x_with_bias, W)
    return int(pred > 0.5)

lin_preds = [linear_predict(x) for x in X_test]

acc = sum(lin_preds == y_test) / len(y_test)
print("Linear Regression Accuracy:", acc)
