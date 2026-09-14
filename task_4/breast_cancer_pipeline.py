"""
Supervised Learning Model Implementation — Breast Cancer Classification
All-in-one script: load -> explore -> feature engineering -> train/test
split -> model training (Logistic Regression, Random Forest, SVM) ->
cross-validation -> hyperparameter tuning -> evaluation -> feature
importance.

Run: python breast_cancer_pipeline.py
Requires: pandas, numpy, matplotlib, seaborn, scikit-learn
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, RocCurveDisplay,
)

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
RANDOM_STATE = 42

RAW_PATH = "breast_cancer.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# STEP 1: LOAD & EXPLORE
# ============================================================
print("=" * 70)
print("STEP 1: LOAD & EXPLORE DATA")
print("=" * 70)

df = pd.read_csv(RAW_PATH)
# target: 0 = malignant, 1 = benign (as encoded in the source dataset)
print("\nShape:", df.shape)
print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nClass balance:\n", df["target"].value_counts())
print("Class balance (%):\n", (df["target"].value_counts(normalize=True) * 100).round(1))

plt.figure(figsize=(5, 4))
sns.countplot(x=df["target"].map({0: "Malignant", 1: "Benign"}), hue=df["target"].map({0: "Malignant", 1: "Benign"}), palette="Set2", legend=False)
plt.title("Class Distribution")
plt.xlabel("Diagnosis")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_class_distribution.png", dpi=140)
plt.close()

# A handful of headline features for a quick visual sense of separability
key_features = ["mean radius", "mean texture", "mean concavity", "mean smoothness"]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for ax, col in zip(axes, key_features):
    sns.kdeplot(data=df, x=col, hue=df["target"].map({0: "Malignant", 1: "Benign"}), ax=ax, fill=True, palette="Set2", legend=(col == key_features[0]))
    ax.set_title(col)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_key_feature_distributions.png", dpi=140)
plt.close()


# ============================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)

X_full = df.drop(columns=["target"])
y = df["target"]

# 2.1 Correlation analysis -> the dataset's 30 features include several
# near-duplicates by construction (radius/perimeter/area, each in mean/
# se/worst form), which adds redundancy and can destabilize coefficient-
# based models (e.g. Logistic Regression) without improving accuracy.
corr_matrix = X_full.corr().abs()
plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix, cmap="coolwarm", cbar=True, xticklabels=False, yticklabels=False)
plt.title("Feature Correlation Heatmap (30 raw features)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_full_correlation_heatmap.png", dpi=140)
plt.close()

# Identify and drop one feature from every highly correlated pair (>0.95)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
print(f"\nDropping {len(to_drop)} highly redundant features (pairwise corr > 0.95):")
print(to_drop)

X_reduced = X_full.drop(columns=to_drop)
print(f"\nFeature count: {X_full.shape[1]} -> {X_reduced.shape[1]} after redundancy removal")

# 2.2 Engineered ratio feature: concavity relative to size often carries
# more diagnostic signal than either raw measurement alone (a small but
# highly irregular mass behaves differently from a large smooth one).
X_reduced = X_reduced.copy()
X_reduced["concavity_to_radius_worst"] = df["worst concavity"] / (df["worst radius"] + 1e-6)
print("Added engineered feature: concavity_to_radius_worst")

feature_names = list(X_reduced.columns)
print(f"\nFinal feature set: {len(feature_names)} features")


# ============================================================
# STEP 3: TRAIN / TEST SPLIT & SCALING
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: TRAIN/TEST SPLIT & SCALING")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
print(f"Train class balance:\n{y_train.value_counts(normalize=True).round(3)}")
print(f"Test class balance:\n{y_test.value_counts(normalize=True).round(3)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeatures standardized using StandardScaler (fit on train only, to avoid data leakage)")


# ============================================================
# STEP 4: MODEL COMPARISON VIA CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: MODEL COMPARISON (5-FOLD STRATIFIED CROSS-VALIDATION)")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

cv_results = {}
for name, model in models.items():
    scores = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=scoring)
    cv_results[name] = {m: scores[f"test_{m}"].mean() for m in scoring}
    cv_results[name]["std_accuracy"] = scores["test_accuracy"].std()
    print(f"\n{name}:")
    for m in scoring:
        print(f"  {m:10s}: {scores[f'test_{m}'].mean():.4f} (+/- {scores[f'test_{m}'].std():.4f})")

cv_df = pd.DataFrame(cv_results).T.round(4)
print("\nCross-validation summary:\n", cv_df)

# Bar chart comparing models
plt.figure(figsize=(8, 5))
cv_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar", ax=plt.gca(), colormap="Set2")
plt.title("5-Fold Cross-Validation: Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0.85, 1.0)
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_model_comparison_cv.png", dpi=140)
plt.close()


# ============================================================
# STEP 5: HYPERPARAMETER TUNING (best-performing model family)
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: HYPERPARAMETER TUNING — Logistic Regression")
print("=" * 70)

# Logistic Regression chosen for tuning: best/near-best CV performance
# AND fully interpretable coefficients -- a meaningful advantage in a
# medical-diagnosis context where clinicians need to trust and audit
# the model's reasoning, not just its accuracy.
param_grid = {"C": [0.01, 0.1, 0.5, 1, 5, 10, 50]}
grid = GridSearchCV(LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
                     param_grid, cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print(f"\nBest params: {grid.best_params_}")
print(f"Best CV F1 score: {grid.best_score_:.4f}")

best_model = grid.best_estimator_


# ============================================================
# STEP 6: FINAL EVALUATION ON HELD-OUT TEST SET
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: FINAL EVALUATION ON TEST SET")
print("=" * 70)

y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
test_prec = precision_score(y_test, y_pred)
test_rec = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_proba)

print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall   : {test_rec:.4f}")
print(f"Test F1 Score : {test_f1:.4f}")
print(f"Test ROC-AUC  : {test_auc:.4f}")
print("\nFull classification report:\n", classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

# Train-set performance (to check for overfitting)
y_train_pred = best_model.predict(X_train_scaled)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"Train Accuracy: {train_acc:.4f} (vs Test Accuracy: {test_acc:.4f}) "
      f"-> gap of {abs(train_acc-test_acc):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
plt.title("Confusion Matrix — Test Set")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_confusion_matrix.png", dpi=140)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5.5, 5))
plt.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC curve (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Test Set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_roc_curve.png", dpi=140)
plt.close()


# ============================================================
# STEP 7: FEATURE IMPORTANCE / INTERPRETABILITY
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: FEATURE IMPORTANCE")
print("=" * 70)

coefs = pd.Series(best_model.coef_[0], index=feature_names).sort_values(key=abs, ascending=False)
print("\nTop 10 features by |coefficient| (standardized scale):\n", coefs.head(10).round(3))

plt.figure(figsize=(8, 6))
top_coefs = coefs.head(12).sort_values()
colors = ["#C44E52" if v < 0 else "#55A868" for v in top_coefs.values]
plt.barh(top_coefs.index, top_coefs.values, color=colors)
plt.axvline(x=0, color="black", linewidth=0.8)
plt.title("Top 12 Features by Logistic Regression Coefficient\n(positive -> pushes toward Benign)")
plt.xlabel("Standardized Coefficient")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_feature_importance.png", dpi=140)
plt.close()

# Save predictions
results_df = X_test.copy()
results_df["actual"] = y_test.values
results_df["predicted"] = y_pred
results_df["predicted_proba_benign"] = y_proba.round(4)
results_df.to_csv("test_predictions.csv", index=False)
print("\nSaved -> test_predictions.csv")
print("\nAll steps complete. Plots saved in ./plots/")
