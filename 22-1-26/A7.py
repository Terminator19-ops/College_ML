import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load thyroid data
data = pd.read_excel("data/data.xlsx", sheet_name="thyroid0387_UCI")

print("Shape:", data.shape)
print(data.head())

# Clean missing values
data = data.replace("?", np.nan).infer_objects(copy=False)

features = data.drop(columns=["Record ID", "Condition"], errors="ignore")

# Encode sex column
if "sex" in features.columns:
    features["sex"] = features["sex"].map({"F": 1, "M": 0})

# Encode binary columns
for col in features.columns:
    if features[col].dtype == "object":
        unique_vals = set(features[col].dropna().unique())
        if unique_vals.issubset({"t", "f"}):
            features[col] = features[col].map({"t": 1, "f": 0})

features = features.apply(pd.to_numeric, errors="coerce")

features = features.fillna(0)

# Get first 20 rows
top20 = features.iloc[:20].values

top20_binary = (top20 > 0).astype(int)

# Similarity functions
def jaccard_similarity(a, b):
    m11 = np.sum((a == 1) & (b == 1))
    m10 = np.sum((a == 1) & (b == 0))
    m01 = np.sum((a == 0) & (b == 1))
    denom = m11 + m10 + m01
    return m11 / denom if denom != 0 else 0

def smc_similarity(a, b):
    m11 = np.sum((a == 1) & (b == 1))
    m00 = np.sum((a == 0) & (b == 0))
    m10 = np.sum((a == 1) & (b == 0))
    m01 = np.sum((a == 0) & (b == 1))
    return (m11 + m00) / (m11 + m00 + m10 + m01)

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b) if (norm_a != 0 and norm_b != 0) else 0

# Compute similarity matrices
n = top20.shape[0]

jaccard_matrix = np.zeros((n, n))
smc_matrix = np.zeros((n, n))
cosine_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        jaccard_matrix[i, j] = jaccard_similarity(top20_binary[i], top20_binary[j])
        smc_matrix[i, j] = smc_similarity(top20_binary[i], top20_binary[j])
        cosine_matrix[i, j] = cosine_similarity(top20[i], top20[j])  

# Create dataframes
labels = [f"V{i+1}" for i in range(n)]

jaccard_df = pd.DataFrame(jaccard_matrix, index=labels, columns=labels)
smc_df = pd.DataFrame(smc_matrix, index=labels, columns=labels)
cosine_df = pd.DataFrame(cosine_matrix, index=labels, columns=labels)

# Visualize heatmaps
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(jaccard_df, annot=True, fmt=".2f", cmap="Blues")
plt.title("Jaccard")

plt.subplot(1, 3, 2)
sns.heatmap(smc_df, annot=True, fmt=".2f", cmap="Greens")
plt.title("SMC")

plt.subplot(1, 3, 3)
sns.heatmap(cosine_df, annot=True, fmt=".2f", cmap="Oranges")
plt.title("Cosine")

plt.tight_layout()
plt.show()
