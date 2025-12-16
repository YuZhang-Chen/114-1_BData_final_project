import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "cleaned_customer_data.csv")
ZIP_PATH = os.path.join(BASE_DIR, "customer_zip.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load data ----------
customer_df = pd.read_csv(DATA_PATH)
zip_df = pd.read_csv(ZIP_PATH, encoding="big5")

# ---------- Basic cleaning ----------
customer_df["Age"] = pd.to_numeric(customer_df["年齡"], errors="coerce")
customer_df["Dependents"] = pd.to_numeric(customer_df["扶養人數"], errors="coerce")
customer_df["Total_Revenue"] = pd.to_numeric(customer_df["總收入"], errors="coerce").fillna(0)

customer_df["Age"].fillna(customer_df["Age"].median(), inplace=True)
customer_df["Dependents"].fillna(customer_df["Dependents"].median(), inplace=True)

customer_df["郵遞區號"] = customer_df["郵遞區號"].astype(str)
zip_df["郵遞區號"] = zip_df["郵遞區號"].astype(str)

# ---------- Merge zipcode data ----------
df = customer_df.merge(zip_df, on="郵遞區號", how="left")

# ---------- CLV ----------
df["CLV"] = df["Total_Revenue"]

# ---------- CLV grouping (30% / 40% / 30%) ----------
low_q = df["CLV"].quantile(0.3)
high_q = df["CLV"].quantile(0.7)

def clv_group(x):
    if x <= low_q:
        return "Low CLV"
    elif x >= high_q:
        return "High CLV"
    else:
        return "Mid CLV"

df["CLV_Group"] = df["CLV"].apply(clv_group)

# ---------- Churn flag ----------
df["Churned"] = df["客戶狀態"].astype(str).str.contains("流失|Churn", regex=True)

# ---------- CLV group summary ----------
clv_summary = df.groupby("CLV_Group").agg(
    Customer_Count=("客戶編號", "count"),
    Average_CLV=("CLV", "mean"),
    Churn_Rate=("Churned", "mean"),
    Average_Age=("Age", "mean"),
    Average_Dependents=("Dependents", "mean")
).reset_index()

clv_summary["Churn_Rate"] = (clv_summary["Churn_Rate"] * 100).round(2)
clv_summary["Average_CLV"] = clv_summary["Average_CLV"].round(2)

print("\n===== CLV Group Summary =====")
print(clv_summary)

# ---------- Plot 1: CLV Distribution ----------
plt.figure()
df["CLV"].plot(kind="hist", bins=40)
plt.title("CLV Distribution")
plt.xlabel("Customer Lifetime Value (CLV)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "clv_distribution.png"), dpi=200)
plt.close()

# ---------- Plot 2: Average CLV by Group ----------
plt.figure()
df.groupby("CLV_Group")["CLV"].mean().reindex(
    ["Low CLV", "Mid CLV", "High CLV"]
).plot(kind="bar")
plt.title("Average CLV by Customer Group")
plt.xlabel("CLV Group")
plt.ylabel("Average CLV")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "avg_clv_by_group.png"), dpi=200)
plt.close()

# ---------- Plot 3: Churn Rate by CLV Group ----------
plt.figure()
(df.groupby("CLV_Group")["Churned"].mean() * 100).reindex(
    ["Low CLV", "Mid CLV", "High CLV"]
).plot(kind="bar")
plt.title("Churn Rate by CLV Group")
plt.xlabel("CLV Group")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "churn_rate_by_group.png"), dpi=200)
plt.close()

# ---------- Plot 4: Average CLV by Promotion ----------
plt.figure(figsize=(10, 5))
df.groupby("優惠方式")["CLV"].mean().sort_values(ascending=False).plot(kind="bar")
plt.title("Average CLV by Promotion Type")
plt.xlabel("Promotion Type")
plt.ylabel("Average CLV")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "promo_vs_clv.png"), dpi=200)
plt.close()

# ---------- Save outputs ----------
df.to_csv(
    os.path.join(os.path.dirname(__file__), "customer_with_clv.csv"),
    index=False,
    encoding="utf-8-sig"
)

clv_summary.to_csv(
    os.path.join(os.path.dirname(__file__), "clv_group_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nAnalysis completed.")
print("Outputs:")
print("09/customer_with_clv.csv")
print("09/clv_group_summary.csv")
print("09/images/*.png")
