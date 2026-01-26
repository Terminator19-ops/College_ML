import pandas as pd
import numpy as np

# Load data
data = pd.read_excel("data/data.xlsx", sheet_name="Purchase data")

features = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
payment = data["Payment (Rs)"].values

# Create binary labels
labels = np.where(payment > 200, 1, 0).reshape(-1, 1)

# Train classifier
pseudo_inverse = np.linalg.pinv(features)
weights = pseudo_inverse @ labels  

print("Weights:\n", weights)

# Predict labels
scores = features @ weights  
predicted = np.where(scores >= 0.5, 1, 0)

predicted_status = np.where(predicted == 1, "RICH", "POOR")
actual_status = np.where(labels == 1, "RICH", "POOR")

# Evaluate accuracy
accuracy = np.mean(predicted == labels)
print("\nAccuracy:", accuracy)

# Display results
results = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)"]].copy()
results["Actual"] = actual_status
results["Predicted"] = predicted_status
results["Score"] = scores

print("\nFirst 10:")
print(results.head(10))
