import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

iris = load_iris()
X = iris.data[iris.target != 2] # type:ignore
y = iris.target[iris.target != 2].astype(float) # type:ignore

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

reg = LinearRegression().fit(X_train, y_train)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mape = mape(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mape = mape(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("All Features LinearRegression")
print()
print("Train MSE :", train_mse)
print("Train RMSE:", train_rmse)
print("Train MAPE:", train_mape)
print("Train R2  :", train_r2)
print()
print("Test MSE :", test_mse)
print("Test RMSE:", test_rmse)
print("Test MAPE:", test_mape)
print("Test R2  :", test_r2)
