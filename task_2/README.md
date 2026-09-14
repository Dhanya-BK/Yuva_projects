## Data Preprocessing & Feature Engineering Workflow

Raw real-world datasets often contain missing values, structural noise, and unformatted attributes that require cleaning before exploratory analysis. The dataset underwent a three-stage preprocessing pipeline: **Missing Value Imputation**, **Feature Selection**, and **Feature Engineering**.

### 1. Handling Missing Data (Imputation Strategy)

The initial audit identified missing values across three primary features: `Age` (177 missing), `Embarked` (2 missing), and `Cabin` (687 missing).

* **`Age` (Numeric Imputation):**  
  `Age` was missing across 19.87% of passenger records. Because the age distribution exhibits slight right-skewness, mean imputation would introduce artificial bias. **Median imputation** was applied ($28.0$ years) to preserve central tendency without distortion from high-age outliers.

* **`Embarked` (Categorical Imputation):**  
  `Embarked` (port of embarkation) was missing in only 2 records ($0.22\%$). **Mode imputation** was used to fill both missing entries with `'S'` (Southampton), the most frequent port ($72.4\%$ of total passengers).

---

### 2. Feature Removal & Selection

* **`Cabin` Column Drop:**  
  `Cabin` was missing over $77\%$ of its data ($687$ out of $891$ records). Imputing over three-quarters of a column creates synthetic bias, so the attribute was dropped entirely.

* **Identifier Exclusion:**  
  High-cardinality metadata such as `PassengerId`, `Name`, and `Ticket` were excluded from correlation matrices and visual aggregations to eliminate noise.

---

### 3. Feature Engineering & Data Transformations

To extract deeper insights from existing columns, synthetic features and categorical mappings were engineered:

* **Family Size Aggregation (`FamilySize`):**  
  The raw dataset separates relatives into `SibSp` (siblings/spouses) and `Parch` (parents/children). A unified metric was generated to track overall group size:
  $$\text{FamilySize} = \text{SibSp} + \text{Parch} + 1$$
  *(The $+1$ accounts for the passenger themselves).*

* **Solo Traveler Indicator (`IsAlone`):**  
  A binary indicator was created to evaluate whether traveling without family impacted survival odds:
  $$\text{IsAlone} = \begin{cases} 1 & \text{if } \text{FamilySize} = 1 \\ 0 & \text{if } \text{FamilySize} > 1 \end{cases}$$

* **Explicit Label Mapping:**  
  Numeric binary flags were converted into descriptive string categories (`0` / `1` $\rightarrow$ `'Did Not Survive'` / `'Survived'`, `1`, `2`, `3` $\rightarrow$ `'1st Class'`, `'2nd Class'`, `'3rd Class'`). This ensures clean annotation in generated visual plots and summary tables.
