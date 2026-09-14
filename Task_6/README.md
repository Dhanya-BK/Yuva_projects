Customer Churn Prediction and Segmentation Using Python
------------------------------------------------------------
Week 6 – Integrative Capstone Project
---------------------------------------
This project demonstrates an end-to-end Data Science workflow using Python. It combines supervised machine learning for customer churn prediction with unsupervised learning for customer segmentation.

Objectives
-------------
Acquire and preprocess a public customer dataset.

Perform exploratory data analysis (EDA).

Predict whether a customer is likely to churn.

Compare Logistic Regression and Random Forest.

Evaluate models using Accuracy, Precision, Recall, F1-score and ROC-AUC.

Segment customers using K-Means clustering.

Derive business insights and recommendations.

Dataset
---------
The project uses the Telco Customer Churn dataset, a commonly used public dataset containing customer demographics, services, account information and churn status.

Download the CSV and place it in the project folder with the filename:

WA_Fn-UseC_-Telco-Customer-Churn.csv

Technologies
-------------
Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Project Structure
------------------------
customer-churn-capstone/
│
├── customer_churn_capstone.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── 01_churn_distribution.png
├── 02_tenure_churn.png
├── 03_monthly_charges_churn.png
├── 04_contract_churn.png
├── 05_model_comparison.png
├── 06_confusion_matrix.png
├── 07_roc_curves.png
├── 08_elbow_method.png
├── 09_silhouette_scores.png
├── 10_customer_segments.png
├── README.md
└── Week6_Capstone_Report.docx

The PNG files are generated automatically when the Python script is executed.

Installation
-----------------
pip install pandas numpy matplotlib seaborn scikit-learn

How to Run
----------
Download the dataset.

Place the CSV file in the same folder as the Python script.

Open a terminal in the project folder.

Run:
-------
python customer_churn_capstone.py

The script prints model metrics and customer-segment summaries and saves the visualizations.

Methodology
------------
Data Collection
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Preprocessing
      ↓
Train/Test Split
      ↓
┌─────────────────────┬─────────────────────┐
│ Supervised Learning │ Unsupervised Learning│
│ Logistic Regression │ K-Means Clustering  │
│ Random Forest       │ Customer Segments   │
└─────────────────────┴─────────────────────┘
      ↓
Evaluation
      ↓
Insights & Recommendations

Models

Logistic Regression

Used as an interpretable baseline classification model.

Random Forest

Used to capture non-linear relationships and compare performance with Logistic Regression.

K-Means

Used to create customer segments based on tenure, monthly charges and total charges.

Evaluation Metrics
-----------------------
Accuracy: Overall proportion of correct predictions.

Precision: Proportion of predicted churners who actually churned.

Recall: Proportion of actual churners correctly identified.

F1-score: Harmonic mean of precision and recall.

ROC-AUC: Measures the model's ability to distinguish churners from non-churners.

Business Insights
-----------------------
The analysis can help identify customer characteristics associated with churn and create customer groups with different tenure and spending patterns. These insights can support targeted retention campaigns rather than applying the same strategy to every customer.

Recommendations
----------------
Prioritize high-risk customers for retention campaigns.

Pay particular attention to customers with short tenure and potentially higher churn risk.

Use contract and service information when designing retention offers.

Combine churn probability with customer segment information to prioritize business actions.

Periodically retrain the model as customer behavior changes.

Limitations
-----------------
The dataset is historical and may not represent current customer behavior.

Model performance depends on the quality and representativeness of the dataset.

K-Means segments depend on the selected features and number of clusters.

Correlation or predictive importance should not automatically be interpreted as causation.

Future Scope
--------------
Hyperparameter tuning and cross-validation.

Try Gradient Boosting/XGBoost if permitted.

Use SHAP or feature importance for explainability.

Deploy the model as a Streamlit web application.

Connect the model to regularly updated customer data.
