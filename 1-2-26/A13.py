import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def confusion_matrix_own(y_true, y_pred):
    TP = TN = FP = FN = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            TP += 1
        elif yt == 0 and yp == 0:
            TN += 1
        elif yt == 0 and yp == 1:
            FP += 1
        elif yt == 1 and yp == 0:
            FN += 1

    return TP, TN, FP, FN

def accuracy_own(TP, TN, FP, FN):
    total = TP + TN + FP + FN
    return (TP + TN) / total

def precision_own(TP, FP):
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def recall_own(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def f1_score_own(precision, recall):
    return (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

TP, TN, FP, FN = confusion_matrix_own(y_test, y_pred)

acc = accuracy_own(TP, TN, FP, FN)
prec = precision_own(TP, FP)
rec = recall_own(TP, FN)
f1 = f1_score_own(prec, rec)

print("Confusion Matrix:")
print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)

print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
