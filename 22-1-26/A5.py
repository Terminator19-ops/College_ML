import pandas as pd
import numpy as np

# Load thyroid data
data = pd.read_excel("data/data.xlsx", sheet_name="thyroid0387_UCI")

first_two_rows = data.iloc[0:2].copy()

# Identify binary columns
binary_cols = []
for col in data.columns:
    unique_vals = set(data[col].dropna().unique())
    if unique_vals.issubset({'t', 'f', '?'}):
        binary_cols.append(col)

binary_data = first_two_rows[binary_cols].copy()

# Clean data
binary_data = binary_data.replace('?', np.nan)

for col in binary_data.columns:
    binary_data[col] = binary_data[col].map({'t': 1, 'f': 0})

binary_data = binary_data.dropna(axis=1)

vec1 = binary_data.iloc[0].to_numpy(dtype=int)
vec2 = binary_data.iloc[1].to_numpy(dtype=int)

# Count matches
f11 = f10 = f01 = f00 = 0

for i, j in zip(vec1, vec2):
    if i == 1 and j == 1:
        f11 += 1
    elif i == 1 and j == 0:
        f10 += 1
    elif i == 0 and j == 1:
        f01 += 1
    elif i == 0 and j == 0:
        f00 += 1

# Calculate coefficients
jaccard = f11 / (f11 + f10 + f01)
smc = (f11 + f00) / (f11 + f10 + f01 + f00)

print("Binary cols detected:", len(binary_cols))
print("Binary cols used:", binary_data.shape[1])
print("f11=", f11, "f10=", f10, "f01=", f01, "f00=", f00)

print("\nJaccard:", jaccard)
print("SMC:", smc)
