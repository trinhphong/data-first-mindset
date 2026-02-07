import pandas as pd
import numpy as np

# Creating a dummy dataset (Imagine this is from a 100MB CSV)
data = {
    "Date": pd.date_range(start="2024-01-01", periods=10),
    "Symbol": [
        "AAPL",
        "AAPL",
        "MSFT",
        "MSFT",
        "GOOG",
        "GOOG",
        "AAPL",
        "MSFT",
        "GOOG",
        "AAPL",
    ],
    "Price": [150.0, 152.5, 300.0, np.nan, 2800.0, 2810.0, 155.0, 305.0, 2790.0, 158.0],
    "Volume": [1000, 1200, 800, 900, 500, 600, 1100, 850, 550, 1300],
}

df = pd.DataFrame(data)

print(df)

# Clean the data
df["Price"] = df["Price"].fillna(df["Price"].mean())

df["Rolling_Avg"] = df["Price"].rolling(window=2).mean()
print(df)

summary = df.groupby("Symbol")["Price"].mean()
print(summary)

# Try to modify the script to find only the days where the Volume was greater than 500 AND the Price was above the average price of the whole dataset.
avg_price = df["Price"].mean()
found_items = df[(df["Volume"] > 500) & (df["Price"] > avg_price)]
print("================")
print("Avg Price:", avg_price)
print("Found Items:")
print(found_items)
print("================")
