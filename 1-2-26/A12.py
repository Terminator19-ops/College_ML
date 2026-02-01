import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def confusion_matrix(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

iris = load_iris()
X = iris.data[iris.target != 2] 
y = iris.target[iris.target != 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
preds = neigh.predict(X_test)

TP, TN, FP, FN = confusion_matrix(y_test, preds)

print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
