import pandas as pd
import numpy as np

# Load data
data = pd.read_excel("data/data.xlsx", sheet_name="Purchase data")
features = data[["Candies (#)","Mangoes (Kg)","Milk Packets (#)"]].values
payment = data["Payment (Rs)"].values.reshape(-1,1)

# Calculate rank
matrix_rank = np.linalg.matrix_rank(features)
print(matrix_rank)

# Compute costs
pseudo_inverse = np.linalg.pinv(features)
costs = pseudo_inverse @ payment

print("Candy:", costs[0][0])
print("Mango:", costs[1][0])
print("Milk:", costs[2][0])