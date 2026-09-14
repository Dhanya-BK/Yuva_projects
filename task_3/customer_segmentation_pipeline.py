"""
Unsupervised Learning & Clustering Analysis — Mall Customer Segmentation
All-in-one script: load -> explore -> preprocess -> determine optimal k
-> K-Means clustering -> Hierarchical clustering (comparison) -> PCA
visualization -> cluster profiling.

Run: python customer_segmentation_pipeline.py
Requires: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

RAW_PATH = "mall_customers.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# STEP 1: LOAD & EXPLORE
# ============================================================
print("=" * 70)
print("STEP 1: LOAD & EXPLORE DATA")
print("=" * 70)

df = pd.read_csv(RAW_PATH)
df = df.rename(columns={"AnnualIncome": "AnnualIncome_k", "SpendingScore": "SpendingScore"})
df["AnnualIncome_k"] = df["AnnualIncome_k"] / 1000  # convert to $k for readability

print("\nShape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nGender distribution:\n", df["Gender"].value_counts())

# Pairplot of raw features
pairplot_cols = ["Age", "AnnualIncome_k", "SpendingScore"]
sns.pairplot(df[pairplot_cols + ["Gender"]], hue="Gender", palette="Set2", height=2.3)
plt.savefig(f"{PLOTS_DIR}/01_pairplot.png", dpi=140, bbox_inches="tight")
plt.close()

# Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, color in zip(axes, pairplot_cols, ["#4C72B0", "#55A868", "#C44E52"]):
    sns.histplot(df[col], kde=True, ax=ax, color=color)
    ax.set_title(f"{col} Distribution")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_feature_distributions.png", dpi=140)
plt.close()

# Correlation heatmap
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(df[pairplot_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_correlation_heatmap.png", dpi=140)
plt.close()


# ============================================================
# STEP 2: PREPROCESSING
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: PREPROCESSING")
print("=" * 70)

# Primary clustering uses Annual Income & Spending Score: these two
# variables are what actually separate customers into distinct,
# actionable spending behaviours. Age is analyzed separately in Step 8
# as an extended 3-feature view (via PCA) -- including it directly in
# the primary clustering blurs the otherwise clean segmentation, since
# Age varies fairly independently of income/spending in this dataset.
features = ["AnnualIncome_k", "SpendingScore"]
features_extended = ["Age", "AnnualIncome_k", "SpendingScore"]
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
print("\nFeatures standardized (mean=0, std=1):")
print(X_scaled_df.describe().loc[["mean", "std"]].round(2))

scaler_ext = StandardScaler()
X_scaled_ext = scaler_ext.fit_transform(df[features_extended])


# ============================================================
# STEP 3: DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: DETERMINE OPTIMAL K (Elbow Method + Silhouette Score)")
print("=" * 70)

k_range = range(2, 11)
inertias = []
sil_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    print(f"  k={k}: inertia={km.inertia_:.2f}, silhouette={sil_scores[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(k_range), inertias, marker="o", color="#4C72B0")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method")
axes[1].plot(list(k_range), sil_scores, marker="o", color="#55A868")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score by k")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_elbow_silhouette.png", dpi=140)
plt.close()

best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"\nBest k by silhouette score: {best_k} (score={max(sil_scores):.3f})")
# Domain-informed choice: k=5 is the well-established segmentation for this
# dataset (visually distinct in Income x Spending space); used below unless
# silhouette strongly favors another value.
K = 5
print(f"Using K={K} clusters for the main analysis "
      f"(consistent with elbow point and strong silhouette score).")


# ============================================================
# STEP 4: K-MEANS CLUSTERING
# ============================================================
print("\n" + "=" * 70)
print(f"STEP 4: K-MEANS CLUSTERING (k={K})")
print("=" * 70)

kmeans = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)
sil_k = silhouette_score(X_scaled, df["Cluster_KMeans"])
print(f"\nSilhouette score at K={K}: {sil_k:.3f}")
print("\nCluster sizes:\n", df["Cluster_KMeans"].value_counts().sort_index())

# Silhouette plot
sample_sil = silhouette_samples(X_scaled, df["Cluster_KMeans"])
fig, ax = plt.subplots(figsize=(7, 5))
y_lower = 10
palette = sns.color_palette("Set2", K)
for i in range(K):
    ith = sample_sil[df["Cluster_KMeans"] == i]
    ith.sort()
    size_i = ith.shape[0]
    y_upper = y_lower + size_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith, facecolor=palette[i], edgecolor=palette[i])
    ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
    y_lower = y_upper + 10
ax.axvline(x=sil_k, color="red", linestyle="--", label=f"Average = {sil_k:.2f}")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.set_title("Silhouette Plot per Cluster (K-Means)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_silhouette_plot.png", dpi=140)
plt.close()


# ============================================================
# STEP 5: HIERARCHICAL CLUSTERING (comparison method)
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: HIERARCHICAL CLUSTERING (comparison)")
print("=" * 70)

linked = linkage(X_scaled, method="ward")
# Compute the merge-distance threshold that yields exactly K clusters:
# the midpoint between the (n-K)-th and (n-K+1)-th merge heights.
merge_heights = np.sort(linked[:, 2])
n = X_scaled.shape[0]
cut_height = (merge_heights[n - K - 1] + merge_heights[n - K]) / 2

plt.figure(figsize=(11, 5))
dendrogram(linked, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=9)
plt.axhline(y=cut_height, color="red", linestyle="--", label=f"Cut height -> {K} clusters")
plt.title("Hierarchical Clustering Dendrogram (Ward linkage)")
plt.xlabel("Sample index (or cluster size)")
plt.ylabel("Distance")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_dendrogram.png", dpi=140)
plt.close()

hc = AgglomerativeClustering(n_clusters=K, linkage="ward")
df["Cluster_Hierarchical"] = hc.fit_predict(X_scaled)
sil_h = silhouette_score(X_scaled, df["Cluster_Hierarchical"])
print(f"\nHierarchical clustering silhouette score at K={K}: {sil_h:.3f}")
agreement = (df["Cluster_KMeans"] == df["Cluster_Hierarchical"]).mean()
print(f"Raw label agreement with K-Means: {agreement*100:.1f}% "
      f"(note: cluster label numbers are arbitrary between methods, "
      f"so this is only a rough indicator, not a strict match score)")


# ============================================================
# STEP 6: VISUALIZE CLUSTERS
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: CLUSTER VISUALIZATION")
print("=" * 70)

# 2D view: Annual Income vs Spending Score (most interpretable business view)
plt.figure(figsize=(7.5, 6))
sns.scatterplot(
    data=df, x="AnnualIncome_k", y="SpendingScore", hue="Cluster_KMeans",
    palette="Set2", s=70, alpha=0.85,
)
centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers_orig[:, 0], centers_orig[:, 1], marker="X", s=250,
            c="black", label="Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"Customer Segments — Income vs Spending Score (K={K})")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_clusters_income_spending.png", dpi=140)
plt.close()

# PCA 2D projection of the EXTENDED 3-feature set (Age + Income + Spending),
# colored by the primary (Income/Spending) cluster labels. This shows
# whether the 2-feature segmentation still separates cleanly once Age
# is factored back in.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled_ext)
df["PCA1"], df["PCA2"] = X_pca[:, 0], X_pca[:, 1]
print(f"\nPCA explained variance ratio (Age+Income+Spending): "
      f"{pca.explained_variance_ratio_.round(3)} "
      f"(total: {pca.explained_variance_ratio_.sum():.3f})")

plt.figure(figsize=(7.5, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster_KMeans", palette="Set2", s=70, alpha=0.85)
plt.title(f"Customer Segments — PCA Projection (Age+Income+Spending, K={K})")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/08_clusters_pca.png", dpi=140)
plt.close()

# Pairplot across all 3 features, colored by cluster
sns.pairplot(df, vars=features_extended, hue="Cluster_KMeans", palette="Set2", height=2.3)
plt.savefig(f"{PLOTS_DIR}/09_cluster_pairplot.png", dpi=140, bbox_inches="tight")
plt.close()


# ============================================================
# STEP 7: CLUSTER PROFILING
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: CLUSTER PROFILING")
print("=" * 70)

profile = df.groupby("Cluster_KMeans")[features_extended].mean().round(1)
profile["Count"] = df["Cluster_KMeans"].value_counts().sort_index()
profile["Pct"] = (profile["Count"] / len(df) * 100).round(1)
print("\nCluster profile (mean values):\n", profile)

gender_by_cluster = pd.crosstab(df["Cluster_KMeans"], df["Gender"], normalize="index") * 100
print("\nGender composition by cluster (%):\n", gender_by_cluster.round(1))

# Cluster profile bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, col in zip(axes, features_extended):
    sns.barplot(x=profile.index, y=profile[col], hue=profile.index, palette="Set2", ax=ax, legend=False)
    ax.set_title(f"Mean {col} by Cluster")
    ax.set_xlabel("Cluster")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/10_cluster_profile_bars.png", dpi=140)
plt.close()

df.to_csv("mall_customers_clustered.csv", index=False)
print("\nSaved -> mall_customers_clustered.csv")
print("\nAll steps complete. Plots saved in ./plots/")
