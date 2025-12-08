# 02.py — Q2: Offer Type Analysis (Fully English, No Chinese, No Font Issues)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------
df = pd.read_csv("cleaned_customer_data.csv")

# Rename columns to English
df = df.rename(columns={
    "優惠方式": "Offer",
    "性別": "Gender",
    "總收入": "Income",
    "婚姻": "MaritalStatus",
    "扶養人數": "Dependents"
})

# -------------------------------------------------------------
# 2. Convert all Offer names into strictly-English labels
# -------------------------------------------------------------
offer_map = {
    "無": "No Offer",
    "無優惠": "No Offer",
    "沒優惠": "No Offer",
    "優惠A": "Offer A",
    "優惠Ｂ": "Offer B",  # in case of full-width
    "優惠B": "Offer B",
    "優惠C": "Offer C",
    "優惠D": "Offer D",
    "優惠E": "Offer E",
    "A": "Offer A",
    "B": "Offer B",
    "C": "Offer C",
    "D": "Offer D",
    "E": "Offer E"
}

# Apply mapping
df["Offer"] = df["Offer"].replace(offer_map)

# If any unknown Offer names remain (e.g., Chinese), map them to "Other"
df["Offer"] = df["Offer"].apply(lambda x: x if str(x).startswith("Offer") or x == "No Offer" else "Other")


# -------------------------------------------------------------
# 3(a). Highest income offer type by gender
# -------------------------------------------------------------
income_by_gender_offer = (
    df.groupby(["Gender", "Offer"])["Income"]
      .mean()
      .reset_index()
      .sort_values(by=["Gender", "Income"], ascending=[True, False])
)

print("=== Highest Average Income Offer by Gender ===")
print(income_by_gender_offer.groupby("Gender").head(1))

plt.figure(figsize=(10, 5))
sns.barplot(data=income_by_gender_offer, x="Offer", y="Income", hue="Gender")
plt.title("Average Income by Offer Type and Gender")
plt.xlabel("Offer Type")
plt.ylabel("Average Income")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -------------------------------------------------------------
# 3(b). Compare characteristics by offer type
# -------------------------------------------------------------
summary_table = df.groupby("Offer")[["MaritalStatus", "Dependents"]].agg({
    "MaritalStatus": lambda x: x.value_counts(normalize=True),
    "Dependents": "mean"
})

print("\n=== Characteristics Comparison by Offer Type ===")
print(summary_table)


# Boxplot for Dependents
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Offer", y="Dependents")
plt.title("Dependents Distribution by Offer Type")
plt.xlabel("Offer Type")
plt.ylabel("Dependents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Marital Status Countplot
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Offer", hue="MaritalStatus")
plt.title("Marital Status Distribution by Offer Type")
plt.xlabel("Offer Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nAnalysis Completed.")
