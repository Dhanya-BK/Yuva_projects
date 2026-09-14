import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.titlesize': 14})

# 1. Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# --- Initial Data Inspection ---
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics (Numerical):\n", df.describe())

# 2. Data Cleaning & Transformations
# Impute Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True) # Dropped due to ~77% missing data

# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Categorical mapping for clarity in plots
df['Survived_Label'] = df['Survived'].map({0: 'Did Not Survive', 1: 'Survived'})
df['Pclass_Label'] = df['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})

# ---------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------

# Figure 1: Target Variable Distribution (Overall Survival)
plt.figure(figsize=(6, 4))
ax1 = sns.countplot(data=df, x='Survived_Label', palette=['#e74c3c', '#2ecc71'])
plt.title('Overall Passenger Survival Distribution', pad=15, weight='bold')
plt.xlabel('Survival Status')
plt.ylabel('Passenger Count')
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 3),
                 textcoords='offset points')
plt.tight_layout()
plt.savefig('fig1_survival_dist.png', dpi=300)
plt.show()

# Figure 2: Survival by Gender and Passenger Class
plt.figure(figsize=(8, 5))
ax2 = sns.barplot(data=df, x='Pclass_Label', y='Survived', hue='Sex', palette='Set2', ci=None)
plt.title('Survival Rate by Passenger Class and Gender', pad=15, weight='bold')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate (0.0 to 1.0)')
plt.legend(title='Gender')
for p in ax2.patches:
    if p.get_height() > 0:
        ax2.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='baseline', fontsize=9, xytext=(0, 3),
                     textcoords='offset points')
plt.tight_layout()
plt.savefig('fig2_survival_class_gender.png', dpi=300)
plt.show()

# Figure 3: Age Distribution across Survival Status
plt.figure(figsize=(9, 5))
sns.kdeplot(data=df, x='Age', hue='Survived_Label', common_norm=False, palette=['#e74c3c', '#2ecc71'], fill=True, alpha=0.4)
plt.title('Age Distribution Density by Survival Status', pad=15, weight='bold')
plt.xlabel('Age (Years)')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('fig3_age_density.png', dpi=300)
plt.show()

# Figure 4: Correlation Heatmap for Numerical Features
plt.figure(figsize=(8, 6))
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features', pad=15, weight='bold')
plt.tight_layout()
plt.savefig('fig4_correlation_heatmap.png', dpi=300)
plt.show()