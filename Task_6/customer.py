# Customer Churn Prediction and Segmentation Using Python
# Week 6 Integrative Capstone Project

# Install if needed:
# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------
# 1. DATA ACQUISITION
# ---------------------------------------------------------
# Download the Telco Customer Churn CSV from IBM/Kaggle/public repositories
# and save it as: WA_Fn-UseC_-Telco-Customer-Churn.csv

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print(df.head())
print(df.info())

# ---------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------
# Customer ID is an identifier, not a predictive feature.
df = df.drop(columns=["customerID"], errors="ignore")

# TotalCharges may contain blank strings; convert them to numeric.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove exact duplicates.
df = df.drop_duplicates()

# Convert target to binary.
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# ---------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------
sns.set_theme(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("01_churn_distribution.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, multiple="stack")
plt.title("Tenure Distribution by Churn")
plt.tight_layout()
plt.savefig("02_tenure_churn.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn")
plt.tight_layout()
plt.savefig("03_monthly_charges_churn.png", dpi=300)
plt.show()

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("04_contract_churn.png", dpi=300)
plt.show()

# Churn rate by selected categorical variables
for col in ["Contract", "InternetService", "PaymentMethod"]:
    rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
    print(f"\nChurn rate by {col}:")
    print((rate * 100).round(2))

# ---------------------------------------------------------
# 4. SUPERVISED LEARNING - PREPROCESSING
# ---------------------------------------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 5. LOGISTIC REGRESSION
# ---------------------------------------------------------
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)
log_pred = logistic_model.predict(X_test)
log_prob = logistic_model.predict_proba(X_test)[:, 1]

print("\nLOGISTIC REGRESSION")
print(classification_report(y_test, log_pred))
print("Accuracy:", accuracy_score(y_test, log_pred))
print("Precision:", precision_score(y_test, log_pred))
print("Recall:", recall_score(y_test, log_pred))
print("F1:", f1_score(y_test, log_pred))
print("ROC-AUC:", roc_auc_score(y_test, log_prob))

# ---------------------------------------------------------
# 6. RANDOM FOREST
# ---------------------------------------------------------
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\nRANDOM FOREST")
print(classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1:", f1_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

# ---------------------------------------------------------
# 7. MODEL COMPARISON
# ---------------------------------------------------------
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, log_pred),
        accuracy_score(y_test, rf_pred)
    ],
    "Precision": [
        precision_score(y_test, log_pred),
        precision_score(y_test, rf_pred)
    ],
    "Recall": [
        recall_score(y_test, log_pred),
        recall_score(y_test, rf_pred)
    ],
    "F1": [
        f1_score(y_test, log_pred),
        f1_score(y_test, rf_pred)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, log_prob),
        roc_auc_score(y_test, rf_prob)
    ]
})

print("\nModel comparison:")
print(results.round(4))

results.set_index("Model").plot(kind="bar", figsize=(10, 5))
plt.title("Supervised Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("05_model_comparison.png", dpi=300)
plt.show()

# Confusion matrix for the better model (change to log_pred if preferred)
best_pred = rf_pred

plt.figure(figsize=(5, 4))
sns.heatmap(
    confusion_matrix(y_test, best_pred),
    annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("06_confusion_matrix.png", dpi=300)
plt.show()

RocCurveDisplay.from_predictions(y_test, log_prob, name="Logistic Regression")
RocCurveDisplay.from_predictions(y_test, rf_prob, name="Random Forest")
plt.title("ROC Curves")
plt.tight_layout()
plt.savefig("07_roc_curves.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 8. UNSUPERVISED LEARNING - CUSTOMER SEGMENTATION
# ---------------------------------------------------------
# Select numeric customer attributes for clustering.
cluster_features = ["tenure", "MonthlyCharges", "TotalCharges"]

cluster_data = df[cluster_features].copy()
cluster_data = cluster_data.fillna(cluster_data.median())

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data)

# Elbow method
inertias = []
silhouette_scores = []
k_values = range(2, 8)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(cluster_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(cluster_scaled, labels))

plt.figure(figsize=(7, 4))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("08_elbow_method.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(list(k_values), silhouette_scores, marker="o")
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("09_silhouette_scores.png", dpi=300)
plt.show()

# Choose k=4 for interpretable customer segmentation.
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(cluster_scaled)

cluster_summary = df.groupby("Cluster")[cluster_features + ["Churn"]].mean()
cluster_summary["ChurnRate"] = cluster_summary["Churn"] * 100

print("\nCustomer segment summary:")
print(cluster_summary.round(2))

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="tenure", y="MonthlyCharges",
    hue="Cluster", palette="tab10", s=60
)
plt.title("Customer Segments: Tenure vs Monthly Charges")
plt.tight_layout()
plt.savefig("10_customer_segments.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 9. FINAL INSIGHTS
# ---------------------------------------------------------
print("\nOverall churn rate:", round(df["Churn"].mean() * 100, 2), "%")
print("\nCluster summary:")
print(cluster_summary.round(2))

print("""
Interpretation:
- Use the supervised model to identify customers who are more likely to churn.
- Recall is important because missing a true churn-risk customer can reduce the
  effectiveness of retention campaigns.
- K-Means provides descriptive customer segments based on tenure and spending.
- Business teams can combine predicted churn risk with segment information to
  prioritize retention actions.
""")
