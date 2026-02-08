import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_points = 20

X_feature = np.random.uniform(1, 10, n_points)
Y_feature = np.random.uniform(1, 10, n_points)
X_train = np.column_stack([X_feature, Y_feature])
y_train = np.where(X_feature + Y_feature > 11, 1, 0)

np.savez('training_data.npz', X=X_train, y=y_train)

plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', s=100, label='Class 0 (Blue)')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', s=100, label='Class 1 (Red)')
plt.xlabel('X Feature')
plt.ylabel('Y Feature')
plt.title('Training Data: 20 Points with 2 Classes')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Class 0 (Blue): {np.sum(y_train == 0)} points")
print(f"Class 1 (Red): {np.sum(y_train == 1)} points")

