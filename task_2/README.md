# Week 2: Exploratory Data Analysis & Visualization (Titanic Dataset)

##  Task Overview
This project focuses on performing **Exploratory Data Analysis (EDA)** and data visualization on the public **Titanic Passenger Manifest dataset**. The goal is to clean raw passenger records, create clear categories, uncover demographic and socio-economic trends, and present findings using Python visualizations and a structured final report.

* **Dataset:** Titanic Passenger Manifest (891 Passengers)
* **Primary Tools:** Python, Pandas, Matplotlib, Seaborn
* **Core Focus:** Data Cleaning, Bivariate Analysis, Feature Engineering, Visual Annotations

  
**1. Missing Value ImputationAge (Numeric):**
---------------------------------------------

1.177 missing entries ($19.87\%$).Replaced using median imputation ($28.0$ years) to prevent right-skewness and extreme elderly outlier bias
2.Embarked (Categorical): 2 missing entries ($0.22\%$). Filled using mode imputation with 'S' (Southampton), the most frequent embarkation port ($72.4\%$).

Feature Selection & RemovalCabin
----------------------------------------
1.Dropped: Omitted entirely due to excessive missingness ($77.10\%$, $687/891$ records) to prevent synthetic imputation bias
2.Identifier Columns: High-cardinality metadata (PassengerId, Name, Ticket) were excluded from numeric correlation calculations.

Feature Engineering & Mappings
------------------------------------
Family Size (FamilySize): Combined isolated relative counts:$$\text{FamilySize} = \text{SibSp} + \text{Parch} + 1$$Solo Traveler Flag (IsAlone): Created a binary indicator:$$\text{IsAlone} = \begin{cases} 1 & \text{if } \text{FamilySize} = 1 \\ 0 & \text{if } \text{FamilySize} > 1 \end{cases}$$Explicit String Labels: Translated raw codes (0/1 $\rightarrow$ 'Did Not Survive'/'Survived'; 1, 2, 3 $\rightarrow$ '1st Class', '2nd Class', '3rd Class') for explicit chart legends and axes.
