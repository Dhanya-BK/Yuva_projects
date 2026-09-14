Titanic Dataset — Data Acquisition, Cleaning & Preprocessing
--------------------------------------------------------------

A single-script pipeline that downloads the Titanic passenger dataset, explores it, detects data quality issues, cleans it, and preprocesses it into an analysis-ready format.

Project Structure
------------------
Task_1/
├── titanic_pipeline.py                      # main script — run this
├── titanic.csv                              # raw input dataset (place here before running)
├── titanic_cleaned.csv                      # output — after cleaning
├── titanic_preprocessed.csv                 # output — after preprocessing
├── Data_Cleaning_Preprocessing_Report.pdf   # full written report
└── plots/                                   # output — generated charts
    ├── 01_missing_heatmap.png
    ├── 02_age_distribution.png
    ├── 03_fare_distribution.png
    ├── 04_correlation_heatmap.png
    └── 05_fare_before_after.png
Requirements
------------------
Python 3.8+
Packages: pandas, numpy, matplotlib, seaborn, scikit-learn

Install them with:
-------------------

bash
pip install pandas numpy matplotlib seaborn scikit-learn
How to Run
----------------
Make sure titanic.csv is in the same folder as titanic_pipeline.py.
Open a terminal in that folder (in VS Code: right-click the file → "Run Python File in Terminal", or cd into the folder manually).
Run:
bash
   python titanic_pipeline.py
The script prints its progress to the console and writes all outputs (CSVs + plots) into the same folder.

What the Script Does
---------------------

Step 1 — Initial Exploration Loads the raw CSV and reports shape, data types, summary statistics, and missing-value counts.

Step 2 — Outlier & Inconsistency Detection Flags statistical outliers in Age, Fare, SibSp, Parch using the IQR method, and detects erroneous entries (e.g. Fare == 0). Saves a missing-value heatmap and distribution plots.

Step 3 — Data Cleaning

Embarked → filled with the most common port (mode)
Age → filled with the median age within each Passenger Class × Title group
Cabin → converted into a HasCabin flag and a Deck category (too sparse to impute directly)
Fare == 0 entries → replaced with the median fare for that class
Extreme Fare outliers → capped (winsorized) rather than removed, so no rows are dropped
Saves titanic_cleaned.csv

Step 4 — Preprocessing

Engineers FamilySize and IsAlone
Drops non-predictive columns (PassengerId, Name, Ticket)
Encodes categorical columns (Sex, Embarked, Title, Deck)
Scales Age, Fare, FamilySize with StandardScaler
Saves titanic_preprocessed.csv (891 rows × 24 columns, fully numeric, no missing values)
Dataset Source

Titanic passenger dataset, public mirror: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

output:
--------------
======================================================================
STEP 1: INITIAL DATA EXPLORATION
======================================================================

Shape: (891, 12)

Data types:
 PassengerId      int64
Survived         int64
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Cabin           object
Embarked        object
dtype: object

First 5 rows:
    PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch  \
0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0   
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0   
2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0   
3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0   
4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0   

             Ticket     Fare Cabin Embarked  
0         A/5 21171   7.2500   NaN        S  
1          PC 17599  71.2833   C85        C  
2  STON/O2. 3101282   7.9250   NaN        S  
3            113803  53.1000  C123        S  
4            373450   8.0500   NaN        S  

Numeric summary:
        PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200

Categorical summary:
                        Name   Sex  Ticket Cabin Embarked
count                   891   891     891   204      889
unique                  891     2     681   147        3
top     Dooley, Mr. Patrick  male  347082    G6        S
freq                      1   577       7     4      644

Missing values:
           missing_count  missing_pct
Age                 177        19.87
Cabin               687        77.10
Embarked              2         0.22

Duplicate rows: 0

======================================================================
STEP 2: OUTLIER & INCONSISTENCY DETECTION
======================================================================

IQR-based outlier counts:
  Age      -> bounds [-6.69, 64.81] | outliers: 11 (1.23%)
  Fare     -> bounds [-26.72, 65.63] | outliers: 116 (13.02%)
  SibSp    -> bounds [-1.50, 2.50] | outliers: 46 (5.16%)
  Parch    -> bounds [0.00, 0.00] | outliers: 213 (23.91%)

Rows with Fare == 0 (inconsistent entries): 15

======================================================================
STEP 3: DATA CLEANING
======================================================================
Filled missing 'Embarked' with mode: 'S'
Imputed 177 missing 'Age' values using Pclass+Title group median
Converted 'Cabin' into 'HasCabin' flag and 'Deck' category
Replaced 15 zero-value 'Fare' entries with class-wise median
Removed 0 duplicate rows
Capped 116 extreme 'Fare' outliers at 66.30

Remaining missing values: 0
Saved -> titanic_cleaned.csv

======================================================================
STEP 4: PREPROCESSING
======================================================================

Final preprocessed shape: (891, 24)
Columns: ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'HasCabin', 'FamilySize', 'IsAlone', 'Embarked_Q', 'Embarked_S', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_T', 'Deck_Unknown']
Saved -> titanic_preprocessed.csv
