import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Step 1: Load the raw dataset
# ----------------------------
df_raw = pd.read_csv("raw_retail_ecommerce_dataset.csv")

# Ensure Invoice_Date is datetime
df_raw["Invoice_Date"] = pd.to_datetime(df_raw["Invoice_Date"])

# ----------------------------
# Step 2: Aggregate daily revenue
# ----------------------------
# Option 1: Actual revenue = Amount Paid
df_daily = df_raw.groupby("Invoice_Date")["Amount_Paid"].sum().reset_index()

# Rename columns for Prophet
df_daily = df_daily.rename(columns={"Invoice_Date": "ds", "Amount_Paid": "y"})

# Sort by date
df_daily = df_daily.sort_values("ds").reset_index(drop=True)

# ----------------------------
# Step 3: Handle missing dates
# ----------------------------
# Create full date range
full_range = pd.date_range(start=df_daily["ds"].min(), end=df_daily["ds"].max())
df_daily = df_daily.set_index("ds").reindex(full_range).fillna(0.0).rename_axis("ds").reset_index()

# ----------------------------
# Step 4: Handle outliers
# ----------------------------
# Define outliers as > mean + 3*std or < mean - 3*std
mean, std = df_daily["y"].mean(), df_daily["y"].std()
upper, lower = mean + 3*std, max(0, mean - 3*std)

# Clip outliers
df_daily["y"] = df_daily["y"].clip(lower=lower, upper=upper)

# ----------------------------
# Step 5: Optional smoothing/feature checks
# ----------------------------
# Rolling average (7-day window) just for visualization
df_daily["y_rolling"] = df_daily["y"].rolling(window=7, min_periods=1).mean()

# ----------------------------
# Step 6: Basic EDA Summary
# ----------------------------
print("Dataset shape:", df_daily.shape)
print("Date range:", df_daily["ds"].min(), "to", df_daily["ds"].max())
print("Missing values:\n", df_daily.isna().sum())
print(df_daily.describe())

# ----------------------------
# Step 7: Visualization
# ----------------------------
plt.figure(figsize=(12,5))
plt.plot(df_daily["ds"], df_daily["y"], label="Daily Revenue", alpha=0.6)
plt.plot(df_daily["ds"], df_daily["y_rolling"], label="7-day Rolling Avg", color="red")
plt.title("Daily Revenue with Rolling Average")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.show()

# ----------------------------
# Step 8: Final cleaned dataset
# ----------------------------
df_final = df_daily[["ds", "y"]]  # this matches your uploaded dataset
df_final.to_csv("financial_forecast_modified.csv", index=False)

print("Cleaned dataset saved as financial_forecast_modified.csv")
