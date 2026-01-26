import pandas as pd
import numpy as np

# Load thyroid data
data = pd.read_excel("data/data.xlsx", sheet_name="thyroid0387_UCI")

print("Shape:", data.shape)
print("\nFirst 5:\n", data.head())

print(data.info())

print(data.columns.tolist())

# Show unique values
for col in data.columns:
    unique_count = data[col].nunique(dropna=True)
    if unique_count <= 15:  
        print(f"\n{col}:")
        print("Unique:", data[col].dropna().unique())

# Identify column types
cat_cols = data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

print("\nCategorical:", cat_cols)
print("Numeric:", num_cols)

print("\nNominal -> One-Hot")
print("Ordinal -> Label")

print("\nCategorical found:")
for col in cat_cols:
    print(f"- {col} (Unique: {data[col].nunique(dropna=True)})")

print("\nNOTE: Check natural order")
print("Ordinal: Low < Med < High")
print("Nominal: Male/Female")

# Numeric ranges
if len(num_cols) > 0:
    ranges = data[num_cols].agg(["min", "max"])
    print(ranges)
else:
    print("No numeric columns")

# Missing values
missing_count = data.isna().sum()
missing_percent = (missing_count / len(data)) * 100

missing_info = pd.DataFrame({
    "Count": missing_count,
    "Percent": missing_percent
}).sort_values(by="Count", ascending=False)

print(missing_info)

# Outlier detection
def detect_outliers(series):
    series = series.dropna()
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return len(outliers), lower, upper

if len(num_cols) > 0:
    for col in num_cols:
        count, low, high = detect_outliers(data[col])
        print(f"{col}: Outliers={count}, Lower={low:.3f}, Upper={high:.3f}")
else:
    print("No numeric columns")

# Statistics
if len(num_cols) > 0:
    stats = pd.DataFrame({
        "Mean": data[num_cols].mean(),
        "Variance": data[num_cols].var(),     
        "Std Dev": data[num_cols].std()
    })
    print(stats)
else:
    print("No numeric columns")
