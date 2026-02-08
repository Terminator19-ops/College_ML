from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

iris = load_iris()
X = iris.data[iris.target != 2] # type: ignore
y = iris.target[iris.target != 2] # type: ignore

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

def evaluate(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name} Results:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    return accuracy_score(y_true, y_pred)

acc_train = evaluate(y_train, y_train_pred, "Training")
acc_test = evaluate(y_test, y_test_pred, "Test")

print(f"\nTrain Acc: {acc_train:.4f}, Test Acc: {acc_test:.4f}")
if acc_train < 0.85 and acc_test < 0.85:
    print("Model is UNDERFITTING")
elif acc_train > 0.95 and acc_test < 0.85:
    print("Model is OVERFITTING")
else:
    print("Model has GOOD FIT")
