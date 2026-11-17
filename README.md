# <Yuva Project> — Data Science & Machine Learning Projects

A collection of six hands-on tasks completed during my internship at **<Company / Program Name>**, covering the full data science workflow: data cleaning, exploratory analysis, clustering, supervised learning, deep learning, and an end-to-end capstone.

## Table of Contents

- [Projects at a Glance](#projects-at-a-glance)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Task 1: Data Acquisition, Cleaning & Preprocessing](#task-1-data-acquisition-cleaning--preprocessing-titanic)
- [Task 2: Exploratory Data Analysis & Visualization](#task-2-exploratory-data-analysis--visualization-titanic)
- [Task 3: Mall Customer Segmentation](#task-3-mall-customer-segmentation-clustering)
- [Task 4: Breast Cancer Classification](#task-4-breast-cancer-classification-supervised-learning)
- [Task 5: Image Classification with CNN](#task-5-image-classification-with-cnn-fashion-mnist)
- [Task 6: Customer Churn Prediction & Segmentation](#task-6-customer-churn-prediction--segmentation-capstone)
- [Tech Stack](#tech-stack)
- [Author](#author)

## Projects at a Glance

| # | Task | Dataset | Techniques | Key Result |
|---|------|---------|------------|------------|
| 1 | Data Acquisition, Cleaning & Preprocessing | Titanic | IQR outlier detection, imputation, encoding, scaling | 891 × 24 fully numeric dataset, 0 missing values |
| 2 | Exploratory Data Analysis & Visualization | Titanic | Bivariate analysis, feature engineering | Cleaned data with `FamilySize` and `IsAlone` features |
| 3 | Customer Segmentation | Mall Customers | K-Means, Hierarchical clustering, PCA | 5 customer segments (silhouette 0.555) |
| 4 | Breast Cancer Classification | Breast Cancer Wisconsin | Logistic Regression, Random Forest, SVM, grid search | 97.4% accuracy, 0.994 ROC-AUC |
| 5 | Image Classification | Fashion-MNIST | 2-block CNN, BatchNorm, Dropout | 89.50% test accuracy |
| 6 | Churn Prediction & Segmentation (Capstone) | Telco Customer Churn | Logistic Regression, Random Forest, K-Means | Model comparison and business recommendations |

## Repository Structure

```text
├── Task_1_Data_Cleaning/
├── Task_2_EDA/
├── Task_3_Clustering/
├── Task_4_Classification/
├── Task_5_CNN/
├── Task_6_Churn_Capstone/
└── README.md
```

Each task folder contains its own script(s), generated plots, and a written report (`.docx`).

## Getting Started

**Prerequisites:** Python 3.8+ and VS Code (or any terminal).

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Core dependencies (used across Tasks 1-4 and 6)
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# Additional dependency for Task 5
pip install tensorflow
```

Then open the folder of the task you want, place the required dataset in it (where noted), and run the main script with `python <script_name>.py`.

---

## Task 1: Data Acquisition, Cleaning & Preprocessing (Titanic)

A single-script pipeline that downloads the Titanic passenger dataset, explores it, detects data quality issues, cleans it, and preprocesses it into an analysis-ready format.

**Run:** place `titanic.csv` next to `titanic_pipeline.py`, then run `python titanic_pipeline.py`.

**What the script does**

1. **Initial exploration:** reports shape, data types, summary statistics, and missing values.
2. **Outlier and inconsistency detection:** flags outliers in `Age`, `Fare`, `SibSp`, `Parch` using the IQR method, and detects erroneous entries (e.g. `Fare == 0`). Saves a missing-value heatmap and distribution plots.
3. **Data cleaning:**
   - `Embarked`: filled with the mode
   - `Age`: filled with the median within each Passenger Class × Title group
   - `Cabin`: converted into a `HasCabin` flag and a `Deck` category
   - `Fare == 0`: replaced with the class-wise median
   - Extreme `Fare` outliers: capped (winsorized) instead of removed, so no rows are dropped
4. **Preprocessing:** engineers `FamilySize` and `IsAlone`, drops non-predictive columns (`PassengerId`, `Name`, `Ticket`), encodes categorical columns, and scales `Age`, `Fare`, `FamilySize` with `StandardScaler`.

**Outputs:** `titanic_cleaned.csv` and `titanic_preprocessed.csv` (891 rows × 24 columns, fully numeric, no missing values).

**Key findings**

| Check | Result |
|-------|--------|
| Missing `Age` | 177 (19.87%) |
| Missing `Cabin` | 687 (77.10%) |
| Missing `Embarked` | 2 (0.22%) |
| Duplicate rows | 0 |
| Rows with `Fare == 0` | 15 |
| Fare outliers capped | 116 (at 66.30) |

**Dataset source:** [Titanic dataset (public mirror)](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

---

## Task 2: Exploratory Data Analysis & Visualization (Titanic)

EDA and visualization on the Titanic Passenger Manifest (891 passengers). The goal is to clean raw records, create clear categories, uncover demographic and socio-economic trends, and present findings through Python visualizations and a structured report.

**Tools:** Python, Pandas, Matplotlib, Seaborn
**Focus:** data cleaning, bivariate analysis, feature engineering, visual annotations

**1. Missing value imputation**
- `Age` (numeric): 177 missing (19.87%). Replaced with the median (28.0 years) to avoid right-skew and elderly-outlier bias.
- `Embarked` (categorical): 2 missing (0.22%). Filled with the mode `S` (Southampton), the most frequent port (72.4%).

**2. Feature selection and removal**
- `Cabin`: dropped due to excessive missingness (77.10%, 687/891 records), avoiding synthetic imputation bias.
- `PassengerId`, `Name`, `Ticket`: high-cardinality identifiers excluded from correlation calculations.

**3. Feature engineering and mappings**
- `FamilySize = SibSp + Parch + 1`
- `IsAlone`: binary flag, `1` if `FamilySize == 1`, else `0`
- Readable labels: `0/1` → `Did Not Survive` / `Survived`; `1, 2, 3` → `1st Class`, `2nd Class`, `3rd Class`, for clearer chart legends and axes.

---

## Task 3: Mall Customer Segmentation (Clustering)

A single-script pipeline that loads the Mall Customer dataset, explores it, finds the optimal number of clusters, applies K-Means and Hierarchical clustering, visualizes the segments, and profiles each cluster.

**Run:** place `mall_customers.csv` next to `customer_segmentation_pipeline.py`, then run `python customer_segmentation_pipeline.py`.

**Folder contents**

```text
├── customer_segmentation_pipeline.py    # main script
├── mall_customers.csv                   # raw input (place here before running)
├── mall_customers_clustered.csv         # output: data with cluster labels + PCA coords
├── Clustering_Analysis_Report.docx      # full written report
└── plots/                               # 10 generated charts (pairplot, elbow, dendrogram, PCA, etc.)
```

**What the script does**

1. **Load and explore:** shape, data types, statistics, missing values, gender distribution; pairplot, distributions, correlation heatmap.
2. **Preprocessing:** selects `AnnualIncome` and `SpendingScore` as clustering features and standardizes them. `Age` is kept aside for descriptive analysis.
3. **Optimal K:** runs K-Means for k = 2 to 10 and plots the Elbow and Silhouette curves, both pointing to K = 5.
4. **K-Means:** K = 5, k-means++ initialization, 10 restarts. Silhouette score: **0.555**.
5. **Hierarchical clustering:** Agglomerative (Ward linkage) with K = 5 plus a dendrogram. Silhouette score: **0.554**, confirming the structure is not specific to K-Means.
6. **Visualization:** Income vs. Spending scatter with centroids, PCA projection, and cluster pairplot.
7. **Profiling:** mean Age, Income, Spending Score and gender mix per cluster.

**Resulting segments (K = 5)**

| Cluster | Persona | Income | Spending | Avg. Age |
|---------|---------|--------|----------|----------|
| 0 | Average / Mainstream | Mid | Mid | ~43 |
| 1 | High-Value Target | High | High | ~33 |
| 2 | Young Aspirational Spenders | Low | High | ~25 |
| 3 | High Income, Low Engagement | High | Low | ~41 |
| 4 | Budget-Conscious | Low | Low | ~45 |

**Dataset source:** [Mall Customers dataset (public mirror)](https://raw.githubusercontent.com/erkansirin78/datasets/master/Mall_Customers.csv)

---

## Task 4: Breast Cancer Classification (Supervised Learning)

A single-script pipeline that loads the Breast Cancer Wisconsin dataset, engineers features, compares three classifiers via cross-validation, tunes the best one, and evaluates it on a held-out test set.

**Run:** place `breast_cancer.csv` next to `breast_cancer_pipeline.py`, then run `python breast_cancer_pipeline.py`.

**Folder contents**

```text
├── breast_cancer_pipeline.py          # main script
├── breast_cancer.csv                  # raw input (place here before running)
├── test_predictions.csv               # output: test-set predictions with probabilities
├── Supervised_Learning_Report.docx    # full written report
└── plots/                             # 7 generated charts
```

**What the script does**

1. **Load and explore:** checks missing values and duplicates, reports class balance, plots key feature distributions by diagnosis.
2. **Feature engineering:** drops 7 highly redundant features (pairwise correlation > 0.95) and adds `concavity_to_radius_worst`, a size-independent shape-irregularity ratio. Final feature set: 24 features (down from 30).
3. **Train/test split and scaling:** 80/20 stratified split, with `StandardScaler` fit only on the training set to avoid data leakage.
4. **Model comparison:** Logistic Regression, Random Forest, and SVM (RBF) using 5-fold stratified cross-validation on accuracy, precision, recall, F1, and ROC-AUC.
5. **Hyperparameter tuning:** grid search over Logistic Regression's regularization strength (`C`), optimizing F1.
6. **Final evaluation:** one-time evaluation on the held-out test set, with confusion matrix, ROC curve, and a train-vs-test overfitting check.
7. **Feature importance:** top 12 features by coefficient magnitude; saves predictions to `test_predictions.csv`.

**Results**

| Metric | Test Score |
|--------|-----------|
| Accuracy | 97.4% |
| Precision | 98.6% |
| Recall | 97.2% |
| F1 Score | 97.9% |
| ROC-AUC | 0.994 |

**Model chosen:** Logistic Regression, which had the best cross-validated performance of the three candidates and is fully interpretable, a real advantage in a medical-diagnosis context.

**Dataset source:** Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository, loaded via scikit-learn's `load_breast_cancer()`.

---

## Task 5: Image Classification with CNN (Fashion-MNIST)

An end-to-end multi-class image classification pipeline built with TensorFlow and Keras that classifies clothing categories from the Fashion-MNIST dataset.

To work within bandwidth and compute limits, a sampled subset of **3,000 training images and 1,000 test images** was used. The custom 2-block CNN uses Batch Normalization and progressive Dropout (25% to 40%) to control overfitting, and trains in about a minute on CPU.

**Folder contents**

```text
├── main.py                              # training and evaluation script
├── generate_report.py                   # builds the Word report
└── Week_05_Deep_Learning_Report.docx    # full project report
```

**Key features**

- **Lightweight dataset:** ~30 MB footprint of 28×28 grayscale apparel images
- **Architecture:** 2-block CNN with 3×3 convolutions, ReLU activations, and Max Pooling
- **Regularization:** Batch Normalization and progressive Dropout (0.25 → 0.40)
- **Optimization:** Adam optimizer with `EarlyStopping` and `ReduceLROnPlateau` callbacks

**Results**

| Metric | Value |
|--------|-------|
| Test Accuracy | 89.50% |
| Test Loss | 0.3206 |

---

## Task 6: Customer Churn Prediction & Segmentation (Capstone)

An end-to-end data science workflow that combines supervised learning for churn prediction with unsupervised learning for customer segmentation.

**Objectives**

- Acquire and preprocess a public customer dataset
- Perform exploratory data analysis
- Predict whether a customer is likely to churn
- Compare Logistic Regression and Random Forest
- Evaluate with Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Segment customers with K-Means
- Derive business insights and recommendations

**Dataset:** Telco Customer Churn. Download the CSV and save it in the project folder as `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

**Folder contents**

```text
├── customer_churn_capstone.py           # main script
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── 01_churn_distribution.png ... 10_customer_segments.png   # generated automatically
└── Week6_Capstone_Report.docx
```

**Run:** `python customer_churn_capstone.py` (prints model metrics and segment summaries, and saves all visualizations).

**Methodology**

```text
Data Collection → Data Cleaning → EDA → Feature Preprocessing → Train/Test Split
        ├── Supervised: Logistic Regression, Random Forest
        └── Unsupervised: K-Means customer segments
→ Evaluation → Insights & Recommendations
```

**Models**

- **Logistic Regression:** interpretable baseline classifier
- **Random Forest:** captures non-linear relationships for comparison
- **K-Means:** segments customers by tenure, monthly charges, and total charges

**Evaluation metrics**

- **Accuracy:** overall share of correct predictions
- **Precision:** share of predicted churners who actually churned
- **Recall:** share of actual churners correctly identified
- **F1-score:** harmonic mean of precision and recall
- **ROC-AUC:** ability to distinguish churners from non-churners

**Business insights and recommendations**

- Prioritize high-risk customers for retention campaigns
- Pay particular attention to short-tenure customers
- Use contract and service information when designing retention offers
- Combine churn probability with customer segment to prioritize actions
- Retrain the model periodically as customer behavior changes

**Limitations**

- The dataset is historical and may not reflect current customer behavior
- Performance depends on data quality and representativeness
- K-Means segments depend on the chosen features and number of clusters
- Predictive importance should not be read as causation

**Future scope:** hyperparameter tuning and cross-validation, Gradient Boosting/XGBoost, SHAP for explainability, a Streamlit deployment, and connection to regularly updated customer data.

---

## Tech Stack

- **Language:** Python
- **Data:** Pandas, NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Deep Learning:** TensorFlow, Keras

## Author

**Dhanya B K**
B.E. in Artificial Intelligence and Machine Learning, Vivekananda College of Engineering and Technology, Puttur
[GitHub](https://github.com/<your-username>)
