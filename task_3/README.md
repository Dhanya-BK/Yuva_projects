# Mall Customer Segmentation — Clustering Analysis

A single-script pipeline that loads the Mall Customer dataset, explores it, determines the optimal number of clusters, applies K-Means and Hierarchical clustering, visualizes the segments, and profiles each cluster.

## Project Structure

```
Task_2/
├── customer_segmentation_pipeline.py   # main script — run this
├── mall_customers.csv                  # raw input dataset (place here before running)
├── mall_customers_clustered.csv        # output — data with cluster labels + PCA coords
├── Clustering_Analysis_Report.docx     # full written report
└── plots/                              # output — generated charts
    ├── 01_pairplot.png
    ├── 02_feature_distributions.png
    ├── 03_correlation_heatmap.png
    ├── 04_elbow_silhouette.png
    ├── 05_silhouette_plot.png
    ├── 06_dendrogram.png
    ├── 07_clusters_income_spending.png
    ├── 08_clusters_pca.png
    ├── 09_cluster_pairplot.png
    └── 10_cluster_profile_bars.png
```

## Requirements

- Python 3.8+
- Packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`

Install them with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## How to Run

1. Make sure `mall_customers.csv` is in the **same folder** as `customer_segmentation_pipeline.py`.
2. Open a terminal **in that folder** (in VS Code: right-click the file → "Run Python File in Terminal", or `cd` into the folder manually).
3. Run:
   ```bash
   python customer_segmentation_pipeline.py
   ```
4. The script prints its progress to the console and writes all outputs (CSV + plots) into the same folder.

## What the Script Does

**Step 1 — Load & Explore**
Loads the raw CSV and reports shape, data types, summary statistics, missing values, and gender distribution. Generates a pairplot, distribution plots, and a correlation heatmap.

**Step 2 — Preprocessing**
Selects `AnnualIncome` and `SpendingScore` as the primary clustering features (the two variables that most clearly separate customers) and standardizes them with `StandardScaler`. Age is kept aside for a secondary, descriptive view later.

**Step 3 — Determine Optimal K**
Runs K-Means for k = 2 to 10, recording inertia (WCSS) and silhouette score at each k. Plots the Elbow curve and Silhouette curve — both point clearly to **K = 5**.

**Step 4 — K-Means Clustering**
Fits K-Means with K=5, `k-means++` initialization, and 10 random restarts. Reports the silhouette score (0.555) and generates a per-cluster silhouette plot.

**Step 5 — Hierarchical Clustering (comparison)**
Fits Agglomerative Clustering (Ward linkage) with K=5 as an independent validation method. Plots a dendrogram with a mathematically-computed cut height. Achieves a near-identical silhouette score (0.554), confirming the 5-cluster structure isn't specific to K-Means.

**Step 6 — Cluster Visualization**
- Income vs. Spending Score scatter plot with centroids (primary, most interpretable view)
- PCA projection of all 3 features (Age + Income + Spending), still colored by cluster
- Pairplot across all features, colored by cluster

**Step 7 — Cluster Profiling**
Computes mean Age, Income, and Spending Score per cluster, plus gender composition, and saves the final labeled dataset as `mall_customers_clustered.csv`.

## Resulting Segments (K=5)

| Cluster | Persona | Income | Spending | Age |
|---|---|---|---|---|
| 0 | Average / Mainstream | Mid | Mid | ~43 |
| 1 | High-Value Target | High | High | ~33 |
| 2 | Young Aspirational Spenders | Low | High | ~25 |
| 3 | High Income, Low Engagement | High | Low | ~41 |
| 4 | Budget-Conscious | Low | Low | ~45 |

## Dataset Source

Mall Customer Segmentation dataset, public mirror:
`https://raw.githubusercontent.com/erkansirin78/datasets/master/Mall_Customers.csv`

## Full Report

See `Clustering_Analysis_Report.docx` for the complete write-up — methodology, rationale behind every choice (features, K, algorithm), cluster interpretations, business implications, and challenges faced.
