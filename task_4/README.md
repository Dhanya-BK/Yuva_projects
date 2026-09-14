# Breast Cancer Classification — Supervised Learning

A single-script pipeline that loads the Breast Cancer Wisconsin dataset, engineers features, compares three classification models via cross-validation, tunes the best one, and evaluates it on a held-out test set.

## Project Structure

```
Task_4/
├── breast_cancer_pipeline.py       # main script — run this
├── breast_cancer.csv               # raw input dataset (place here before running)
├── test_predictions.csv            # output — test-set predictions with probabilities
├── Supervised_Learning_Report.docx # full written report
└── plots/                          # output — generated charts
    ├── 01_class_distribution.png
    ├── 02_key_feature_distributions.png
    ├── 03_full_correlation_heatmap.png
    ├── 04_model_comparison_cv.png
    ├── 05_confusion_matrix.png
    ├── 06_roc_curve.png
    └── 07_feature_importance.png
```

## Requirements

- Python 3.8+
- Packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

Install them with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Make sure `breast_cancer.csv` is in the **same folder** as `breast_cancer_pipeline.py`.
2. Open a terminal **in that folder** (in VS Code: right-click the file → "Run Python File in Terminal", or `cd` into the folder manually).
3. Run:
   ```bash
   python breast_cancer_pipeline.py
   ```
4. The script prints its progress to the console and writes all outputs (CSV + plots) into the same folder.

## What the Script Does

**Step 1 — Load & Explore**
Loads the dataset, checks for missing values/duplicates, reports class balance, and plots key feature distributions split by diagnosis.

**Step 2 — Feature Engineering**
- Drops 7 highly redundant features (pairwise correlation > 0.95) — e.g. mean radius/perimeter/area all measure size and are near-duplicates
- Engineers a new feature: `concavity_to_radius_worst`, a shape-irregularity ratio independent of tumor size
- Final feature set: 24 features (down from 30)

**Step 3 — Train/Test Split & Scaling**
80/20 stratified split (preserves class balance in both sets), then `StandardScaler` fit only on the training set to avoid data leakage.

**Step 4 — Model Comparison (Cross-Validation)**
Compares Logistic Regression, Random Forest, and SVM (RBF kernel) using 5-fold stratified cross-validation across accuracy, precision, recall, F1, and ROC-AUC.

**Step 5 — Hyperparameter Tuning**
Grid search over Logistic Regression's regularization strength (C), optimizing for F1 score.

**Step 6 — Final Evaluation**
Evaluates the tuned model once on the held-out test set: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, and a train-vs-test comparison to check for overfitting.

**Step 7 — Feature Importance**
Extracts and plots the top 12 features by logistic regression coefficient magnitude, and saves all test-set predictions (with probabilities) to `test_predictions.csv`.

## Results Summary

| Metric | Test Score |
|---|---|
| Accuracy | 97.4% |
| Precision | 98.6% |
| Recall | 97.2% |
| F1 Score | 97.9% |
| ROC-AUC | 0.994 |

**Model chosen:** Logistic Regression — best cross-validated performance among the three candidates *and* fully interpretable, a meaningful advantage in a medical-diagnosis context.

## Dataset Source

Breast Cancer Wisconsin (Diagnostic) Dataset, from the UCI Machine Learning Repository, loaded via scikit-learn's built-in `load_breast_cancer()`.

## Full Report

See `Supervised_Learning_Report.docx` for the complete write-up — problem definition, feature engineering rationale, model comparison, hyperparameter tuning, evaluation, feature importance, strengths/limitations, and possible improvements.
