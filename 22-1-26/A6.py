import pandas as pd
import numpy as np

# Load thyroid data
data = pd.read_excel("data/data.xlsx", sheet_name="thyroid0387_UCI")

# Clean missing values
obj_cols = data.select_dtypes(include="object").columns
for col in obj_cols:
    data[col] = data[col].mask(data[col] == "?", np.nan)

# Encode binary columns
for col in obj_cols:
    unique_vals = set(data[col].dropna().unique())
    if unique_vals.issubset({"t", "f"}):
        data[col] = data[col].map({"t": 1, "f": 0})

# One-hot encode
encoded_data = pd.get_dummies(data, drop_first=False)

# Fill missing values
for col in encoded_data.columns:
    if encoded_data[col].dtype != "uint8":
        encoded_data[col] = encoded_data[col].fillna(encoded_data[col].mean())

# Get vectors
vec_a = encoded_data.iloc[0].to_numpy(dtype=float)
vec_b = encoded_data.iloc[1].to_numpy(dtype=float)

# Calculate similarity
cosine_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

print("Cosine Similarity:", cosine_sim)
