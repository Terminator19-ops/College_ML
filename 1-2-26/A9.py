import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Predicted labels for test set:")
print(y_pred)

sample_vector = X_test[0]
predicted_class = knn.predict([sample_vector])

print("Test Vector:", sample_vector)
print("Predicted Class:", predicted_class[0])
print("Actual Class:", y_test[0])