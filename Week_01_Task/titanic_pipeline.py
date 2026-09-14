"""
Data Acquisition, Cleaning, and Preprocessing — Titanic Dataset
All-in-one script: initial exploration -> outlier detection ->
data cleaning -> preprocessing.

Run: python titanic_pipeline.py
Requires: pandas, numpy, matplotlib, seaborn, scikit-learn
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

RAW_PATH = "titanic.csv"
PLOTS_DIR = "plots"
import os
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# STEP 1: INITIAL DATA EXPLORATION
# ============================================================
print("=" * 70)
print("STEP 1: INITIAL DATA EXPLORATION")
print("=" * 70)

df = pd.read_csv(RAW_PATH)

print("\nShape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nNumeric summary:\n", df.describe())
print("\nCategorical summary:\n", df.describe(include=["object"]))

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
print("\nMissing values:\n", missing_report[missing_report.missing_count > 0])

print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# STEP 2: OUTLIER & INCONSISTENCY DETECTION
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: OUTLIER & INCONSISTENCY DETECTION")
print("=" * 70)

# Missing value heatmap
plt.figure(figsize=(9, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="rocket_r", yticklabels=False)
plt.title("Missing Value Map (Titanic Raw Dataset)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_missing_heatmap.png", dpi=140)
plt.close()

# Age & Fare distributions
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(df["Age"].dropna(), kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("Age Distribution")
sns.boxplot(x=df["Age"], ax=axes[1], color="#4C72B0")
axes[1].set_title("Age — Boxplot")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_age_distribution.png", dpi=140)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(df["Fare"], kde=True, ax=axes[0], color="#C44E52")
axes[0].set_title("Fare Distribution (raw)")
sns.boxplot(x=df["Fare"], ax=axes[1], color="#C44E52")
axes[1].set_title("Fare — Boxplot")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_fare_distribution.png", dpi=140)
plt.close()

def iqr_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return mask.sum(), lower, upper

print("\nIQR-based outlier counts:")
for col in ["Age", "Fare", "SibSp", "Parch"]:
    n_out, low, high = iqr_outliers(df[col].dropna())
    print(f"  {col:8s} -> bounds [{low:.2f}, {high:.2f}] | outliers: {n_out} "
          f"({n_out/len(df)*100:.2f}%)")

zero_fare = (df["Fare"] == 0).sum()
print(f"\nRows with Fare == 0 (inconsistent entries): {zero_fare}")

# raw snapshot used later for before/after comparison plots
df_raw_snapshot = df.copy()


# ============================================================
# STEP 3: DATA CLEANING
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: DATA CLEANING")
print("=" * 70)

# 3.1 Embarked -> mode imputation
mode_embarked = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(mode_embarked)
print(f"Filled missing 'Embarked' with mode: '{mode_embarked}'")

# 3.2 Age -> Pclass x Title group median imputation
df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")
title_map = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
    "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
    "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
}
df["Title"] = df["Title"].replace(title_map)
df.loc[~df["Title"].isin(["Mr", "Mrs", "Miss", "Master", "Rare"]), "Title"] = "Rare"

age_missing = df["Age"].isnull().sum()
df["Age"] = df.groupby(["Pclass", "Title"])["Age"].transform(lambda s: s.fillna(s.median()))
df["Age"] = df["Age"].fillna(df["Age"].median())
print(f"Imputed {age_missing} missing 'Age' values using Pclass+Title group median")

# 3.3 Cabin -> HasCabin flag + Deck category
df["HasCabin"] = df["Cabin"].notnull().astype(int)
df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
df = df.drop(columns=["Cabin"])
print("Converted 'Cabin' into 'HasCabin' flag and 'Deck' category")

# 3.4 Zero-fare correction
median_fare_by_class = df[df["Fare"] > 0].groupby("Pclass")["Fare"].median()
for pclass, med in median_fare_by_class.items():
    mask = (df["Fare"] == 0) & (df["Pclass"] == pclass)
    df.loc[mask, "Fare"] = med
print(f"Replaced {zero_fare} zero-value 'Fare' entries with class-wise median")

# 3.5 Duplicates
dupes = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Removed {dupes} duplicate rows")

# 3.6 Fare outlier capping (winsorizing)
q1, q3 = df["Fare"].quantile(0.25), df["Fare"].quantile(0.75)
upper_bound = q3 + 1.5 * (q3 - q1)
n_capped = (df["Fare"] > upper_bound).sum()
df["Fare"] = np.where(df["Fare"] > upper_bound, upper_bound, df["Fare"])
print(f"Capped {n_capped} extreme 'Fare' outliers at {upper_bound:.2f}")

print("\nRemaining missing values:", df.isnull().sum().sum())
df.to_csv("titanic_cleaned.csv", index=False)
print("Saved -> titanic_cleaned.csv")

# Before/after comparison plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(x=df_raw_snapshot["Fare"], ax=axes[0], color="#C44E52")
axes[0].set_title("Fare — Before Cleaning")
sns.boxplot(x=df["Fare"], ax=axes[1], color="#55A868")
axes[1].set_title("Fare — After Outlier Capping")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_fare_before_after.png", dpi=140)
plt.close()


# ============================================================
# STEP 4: PREPROCESSING
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: PREPROCESSING")
print("=" * 70)

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df = df.drop(columns=["PassengerId", "Name", "Ticket"])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked", "Title", "Deck"], drop_first=True)

numeric_cols = ["Age", "Fare", "FamilySize"]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Correlation heatmap (post-cleaning, pre-scaling would also work; using final numeric set)
plt.figure(figsize=(6.5, 5))
corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_correlation_heatmap.png", dpi=140)
plt.close()

print("\nFinal preprocessed shape:", df.shape)
print("Columns:", list(df.columns))

df.to_csv("titanic_preprocessed.csv", index=False)
print("Saved -> titanic_preprocessed.csv")

print("\nAll steps complete. Plots saved in ./plots/")
