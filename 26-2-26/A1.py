import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

iris = load_iris()
X = iris.data[iris.target != 2] # type:ignore
y = iris.target[iris.target != 2].astype(float) # type:ignore

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train_single = X_train[:, 0].reshape(-1, 1)
X_test_single = X_test[:, 0].reshape(-1, 1)

reg = LinearRegression().fit(X_train_single, y_train)
y_train_pred = reg.predict(X_train_single)

print("Coefficient:", reg.coef_)
print("Intercept:", reg.intercept_)
print("First 5 predictions (train):", y_train_pred[:5])
print("First 5 actuals (train):", y_train[:5])
