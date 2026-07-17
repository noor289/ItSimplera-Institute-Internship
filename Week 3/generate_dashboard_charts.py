"""
Generates the 3 static PNG charts shown on the /dashboard page.
Run this once before starting the app: python generate_dashboard_charts.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

df = pd.read_excel("Week 2 (DataSet).xlsx", sheet_name="Steel_industry_data")
df["date"] = pd.to_datetime(df["date"])
df["hour"] = df["date"].dt.hour

# 1. Average energy usage by hour of day
hourly_avg = df.groupby("hour")["Usage_kWh"].mean()
plt.figure(figsize=(8, 5))
hourly_avg.plot(kind="bar", color="#4C72B0")
plt.title("Average Energy Usage by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Average Usage (kWh)")
plt.tight_layout()
plt.savefig(STATIC_DIR / "energy_by_hour.png", dpi=120)
plt.close()

# 2. Average energy usage by load type
load_avg = df.groupby("Load_Type")["Usage_kWh"].mean().sort_values()
plt.figure(figsize=(7, 5))
load_avg.plot(kind="bar", color="#DD8452")
plt.title("Average Energy Usage by Load Type")
plt.xlabel("Load Type")
plt.ylabel("Average Usage (kWh)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(STATIC_DIR / "energy_by_load_type.png", dpi=120)
plt.close()

# 3. Correlation heatmap (numeric columns only)
numeric_cols = df.select_dtypes(include="number").columns
plt.figure(figsize=(9, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(STATIC_DIR / "correlation_heatmap.png", dpi=120)
plt.close()

print("Saved: energy_by_hour.png, energy_by_load_type.png, correlation_heatmap.png -> static/")