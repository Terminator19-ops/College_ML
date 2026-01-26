import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Load stock data
data = pd.read_excel("data/data.xlsx", sheet_name="IRCTC Stock Price")

prices = data.iloc[:, 3].to_numpy()

# Process change column
change_raw = data.iloc[:, 8]

change = change_raw.astype(str).str.replace("%", "", regex=False)
change = pd.to_numeric(change, errors="coerce")
data["Change"] = change

# NumPy stats
mean_numpy = np.mean(prices)
var_numpy = np.var(prices)

print("Mean (NumPy):", mean_numpy)
print("Variance (NumPy):", var_numpy)

# Custom functions
def calculate_mean(arr):
    total = 0.0
    for x in arr:
        total += x
    return total / len(arr)

def calculate_variance(arr):
    m = calculate_mean(arr)
    total = 0.0
    for x in arr:
        total += (x - m) ** 2
    return total / len(arr)  

mean_custom = calculate_mean(prices)
var_custom = calculate_variance(prices)

print("\nMean (Custom):", mean_custom)
print("Variance (Custom):", var_custom)

print("\nMean Diff:", mean_custom - mean_numpy)
print("Variance Diff:", var_custom - var_numpy)

# Timing benchmarks
def time_numpy(arr, runs=10):
    start = time.perf_counter()
    for _ in range(runs):
        np.mean(arr)
        np.var(arr)
    end = time.perf_counter()
    return (end - start) / runs

def time_custom(arr, runs=10):
    start = time.perf_counter()
    for _ in range(runs):
        calculate_mean(arr)
        calculate_variance(arr)
    end = time.perf_counter()
    return (end - start) / runs

print("\nTime (NumPy):", time_numpy(prices))
print("Time (Custom):", time_custom(prices))

# Wednesday analysis
wed_data = data[data["Day"] == "Wed"]
wed_prices = wed_data.iloc[:, 3].to_numpy()

wed_mean = np.mean(wed_prices)

print("\nPopulation Mean:", mean_numpy)
print("Wednesday Mean:", wed_mean)
print("Diff:", mean_numpy - wed_mean)

# April analysis
apr_data = data[data["Month"] == "Apr"]
apr_prices = apr_data.iloc[:, 3].to_numpy()

apr_mean = np.mean(apr_prices)

print("\nApril Mean:", apr_mean)
print("Diff:", mean_numpy - apr_mean)

# Probability calculations
valid_change = data["Change"].dropna()

loss_days = valid_change.apply(lambda x: x < 0).sum()
total_days = len(valid_change)

prob_loss = loss_days / total_days
print("\nP(Loss):", prob_loss)

profit_wed = ((data["Day"] == "Wed") & (data["Change"] > 0)).sum()
prob_profit_wed = profit_wed / total_days
print("\nP(Profit & Wed):", prob_profit_wed)

wed_total = (data["Day"] == "Wed").sum()
wed_profit = ((data["Day"] == "Wed") & (data["Change"] > 0)).sum()

cond_prob = wed_profit / wed_total
print("\nP(Profit|Wed):", cond_prob)

# Scatter plot
day_map = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5}
data["Day_num"] = data["Day"].map(day_map)

plt.figure(figsize=(7, 4))
plt.scatter(data["Day_num"], data["Change"])
plt.xticks([1, 2, 3, 4, 5], ["Mon", "Tue", "Wed", "Thu", "Fri"])
plt.xlabel("Day")
plt.ylabel("Change %")
plt.title("Change vs Day (IRCTC)")
plt.grid(True)
plt.show()
