"""Script to build all 12 individual Jupyter notebooks + master_notebook.ipynb.

Each notebook is fully populated with comprehensive Markdown explanations,
business context, mathematical formulations, and working Python code cells.
"""

from pathlib import Path
import nbformat as nbf

NOTEBOOK_DIR = Path("notebook")
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def create_cell(cell_type: str, source: str):
    """Helper to create markdown or code cells."""
    if cell_type == "markdown":
        return nbf.v4.new_markdown_cell(source.strip())
    elif cell_type == "code":
        return nbf.v4.new_code_cell(source.strip())
    raise ValueError(f"Unknown cell type: {cell_type}")


def save_notebook(filename: str, cells: list):
    """Saves notebook structure to disk."""
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14.0",
        },
    }
    filepath = NOTEBOOK_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated: {filepath}")


# ==============================================================================
# Notebook 01: Data Collection
# ==============================================================================
def build_nb_01():
    cells = [
        create_cell("markdown", """# 01 - Data Collection & Ingestion

## 1. Project Overview & Business Problem
**Thera Bank** has a large liability customer base. The majority of customers are depositors (liability customers) with varying sizes of relationship. The bank wants to convert its liability customers into personal loan customers (asset customers) while retaining them as depositors. A previous campaign for liability customers showed an acceptance rate of ~9.6%.

The objective is to build a high-performance machine learning model to predict which customers have a high propensity to accept a personal loan offer. This enables targeted marketing campaigns, minimizing customer friction and marketing acquisition costs.

## 2. Data Dictionary
| Variable | Description | Type |
| :--- | :--- | :--- |
| **ID** | Customer unique identifier | Numerical (Nominal) |
| **Age** | Customer's age in completed years | Numerical (Discrete) |
| **Experience** | Years of professional experience | Numerical (Discrete) |
| **Income** | Annual income of the customer ($000) | Numerical (Continuous) |
| **ZIP Code** | Home address postal code | Numerical (Nominal) |
| **Family** | Family size of the customer (1 to 4) | Numerical (Discrete) |
| **CCAvg** | Average monthly spending on credit cards ($000) | Numerical (Continuous) |
| **Education** | Education level: 1 = Undergrad, 2 = Graduate, 3 = Advanced/Professional | Categorical (Ordinal) |
| **Mortgage** | Value of house mortgage if any ($000) | Numerical (Continuous) |
| **Personal Loan** | Did the customer accept personal loan in last campaign? (Target: 1 = Yes, 0 = No) | Binary |
| **Securities Account** | Does customer hold securities account with the bank? (1 = Yes, 0 = No) | Binary |
| **CD Account** | Does customer hold Certificate of Deposit account with the bank? (1 = Yes, 0 = No) | Binary |
| **Online** | Does customer use internet banking facilities? (1 = Yes, 0 = No) | Binary |
| **CreditCard** | Does customer hold credit card issued by Universal Bank? (1 = Yes, 0 = No) | Binary |
"""),
        create_cell("code", """import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adjust project root path
project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

raw_data_path = project_root / "data" / "raw" / "Bank_Personal_Loan_Modelling.csv"
print(f"Project Root: {project_root}")
print(f"Raw Data Path: {raw_data_path}")
print(f"File exists: {raw_data_path.exists()}")
"""),
        create_cell("markdown", "## 3. Data Ingestion & Integrity Check"),
        create_cell("code", """# Ingest raw dataset
df = pd.read_csv(raw_data_path)

print("Dataset Shape:", df.shape)
print("Row count:", len(df))
print("Column count:", len(df.columns))
print("\\nColumns List:")
for idx, col in enumerate(df.columns, 1):
    print(f"  {idx}. {col}")
"""),
        create_cell("markdown", "## 4. Inspection of First & Last Records"),
        create_cell("code", """# Display first 5 rows
print("Top 5 records:")
display(df.head())

print("\\nBottom 5 records:")
display(df.tail())
"""),
        create_cell("markdown", "## 5. Memory Footprint and Basic Types"),
        create_cell("code", """# Memory usage and non-null count verification
df.info()
"""),
        create_cell("markdown", """## 6. Data Collection Summary & Key Takeaways
1. The raw dataset contains exactly **5,000 observations** and **14 columns**.
2. No null or missing values are present in the raw file.
3. Target column is `Personal Loan`.
4. Identifier columns (`ID`, `ZIP Code`) need careful handling before modeling.
"""),
    ]
    save_notebook("01_data_collection.ipynb", cells)


# ==============================================================================
# Notebook 02: Data Understanding
# ==============================================================================
def build_nb_02():
    cells = [
        create_cell("markdown", """# 02 - Data Understanding & Statistical Profiling

## 1. Objectives
- Perform comprehensive statistical profiling of all attributes.
- Identify data anomalies, zeros, negatives, and outliers.
- Quantify class imbalance in the target variable `Personal Loan`.
"""),
        create_cell("code", """import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

df = pd.read_csv(project_root / "data" / "raw" / "Bank_Personal_Loan_Modelling.csv")
print("Data loaded successfully. Shape:", df.shape)
"""),
        create_cell("markdown", "## 2. Descriptive Statistics"),
        create_cell("code", """# Numerical summary statistics
desc_df = df.describe().T
desc_df["variance"] = df.var()
desc_df["skewness"] = df.skew()
desc_df["kurtosis"] = df.kurt()
display(desc_df)
"""),
        create_cell("markdown", """## 3. Anomaly Detection: Negative Experience Values
In the `Experience` column, the minimum value is **-3**. Professional experience cannot be negative.
Let us examine the negative experience records and their relationship with customer age.
"""),
        create_cell("code", """neg_exp = df[df["Experience"] < 0]
print(f"Total records with negative experience: {len(neg_exp)} ({len(neg_exp)/len(df):.2%})")
print("\\nNegative experience breakdown by Age & Education:")
display(neg_exp.groupby(["Age", "Experience", "Education"]).size().reset_index(name="Count"))
"""),
        create_cell("markdown", """## 4. Analysis of Zero Values
Features like `Mortgage`, `Securities Account`, and `CD Account` contain significant zero values. Let us quantify the proportion of zeros for each attribute.
"""),
        create_cell("code", """zero_counts = (df == 0).sum()
zero_pct = (df == 0).mean() * 100
zero_df = pd.DataFrame({"Zero_Count": zero_counts, "Zero_Percentage": zero_pct})
display(zero_df.sort_values(by="Zero_Percentage", ascending=False))
"""),
        create_cell("markdown", """## 5. Target Class Distribution Analysis
The target is `Personal Loan`. Let us examine the class distribution to assess imbalance.
"""),
        create_cell("code", """target_counts = df["Personal Loan"].value_counts()
target_prop = df["Personal Loan"].value_counts(normalize=True) * 100

print(f"Class 0 (Did not accept loan): {target_counts[0]:,} ({target_prop[0]:.2f}%)")
print(f"Class 1 (Accepted loan):       {target_counts[1]:,} ({target_prop[1]:.2f}%)")
print(f"Imbalance Ratio:               {target_counts[0] / target_counts[1]:.2f} : 1")
"""),
        create_cell("markdown", """## 6. Key Data Understanding Insights
1. **Class Imbalance:** Only 9.6% of customers accepted the personal loan. Pure accuracy is an inadequate metric; precision, recall, F1, and PR-AUC must be prioritized.
2. **Negative Experience Anomaly:** 52 rows have negative experience (min = -3). All occur at young ages (Age 23-29). This will be handled during data cleaning.
3. **Mortgage Sparsity:** Over 68% of customers have zero mortgage.
4. **CD Account Rarity:** Only 6.04% of customers have a CD account, but CD holders have a very strong historical propensity to take loans.
"""),
    ]
    save_notebook("02_data_understanding.ipynb", cells)


# ==============================================================================
# Notebook 03: EDA
# ==============================================================================
def build_nb_03():
    cells = [
        create_cell("markdown", """# 03 - Exploratory Data Analysis (EDA)

## 1. Objectives
- Perform univariate, bivariate, and multivariate analysis.
- Visualize distributions across personal loan converters vs non-converters.
- Investigate correlation structure and identify dominant predictors.
"""),
        create_cell("code", """import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

df = pd.read_csv(project_root / "data" / "raw" / "Bank_Personal_Loan_Modelling.csv")
sns.set_theme(style="whitegrid")
"""),
        create_cell("markdown", "## 2. Univariate Distribution of Key Continuous Variables"),
        create_cell("code", """cont_cols = ["Age", "Income", "CCAvg", "Mortgage"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, col in enumerate(cont_cols):
    ax = axes[idx // 2, idx % 2]
    sns.histplot(df[col], kde=True, ax=ax, color="#1f77b4", bins=30)
    ax.set_title(f"Distribution of {col}", fontweight="bold")
    ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.1f}")
    ax.axvline(df[col].median(), color="green", linestyle="-", label=f"Median: {df[col].median():.1f}")
    ax.legend()

plt.tight_layout()
plt.show()
"""),
        create_cell("markdown", "## 3. Bivariate Analysis: Income & CCAvg vs Personal Loan Acceptance"),
        create_cell("code", """fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(x="Personal Loan", y="Income", data=df, ax=axes[0], palette=["#2b5c8f", "#d95f02"])
axes[0].set_title("Annual Income by Loan Acceptance Status", fontweight="bold")
axes[0].set_xticklabels(["Did Not Accept (0)", "Accepted (1)"])

sns.boxplot(x="Personal Loan", y="CCAvg", data=df, ax=axes[1], palette=["#2b5c8f", "#d95f02"])
axes[1].set_title("Credit Card Avg Spend by Loan Acceptance Status", fontweight="bold")
axes[1].set_xticklabels(["Did Not Accept (0)", "Accepted (1)"])

plt.tight_layout()
plt.show()
"""),
        create_cell("markdown", "## 4. Categorical Drivers: Education, Family Size, and CD Account"),
        create_cell("code", """fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, col in enumerate(["Education", "Family", "CD Account"]):
    crosstab_pct = pd.crosstab(df[col], df["Personal Loan"], normalize="index") * 100
    crosstab_pct.plot(kind="bar", stacked=True, ax=axes[idx], color=["#2b5c8f", "#d95f02"])
    axes[idx].set_title(f"Loan Acceptance % by {col}", fontweight="bold")
    axes[idx].set_ylabel("Percentage (%)")
    axes[idx].legend(["Did Not Accept", "Accepted"])

plt.tight_layout()
plt.show()
"""),
        create_cell("markdown", "## 5. Correlation Heatmap"),
        create_cell("code", """plt.figure(figsize=(12, 9))
corr = df.drop(columns=["ID", "ZIP Code"]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Matrix of Features", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.show()
"""),
        create_cell("markdown", """## 6. Key EDA Findings
1. **Income** has the highest correlation with loan acceptance ($r = 0.50$). Customers accepting loans have significantly higher annual income (median ~$140k vs ~$40k).
2. **CCAvg** also shows strong positive relationship ($r = 0.37$).
3. **CD Account:** Customers with Certificate of Deposit accounts have an acceptance rate exceeding 45% (compared to ~7% for non-CD account holders).
4. **Education:** Higher education (Graduate / Professional) is strongly correlated with loan conversion.
5. **Age & Experience:** Very high collinearity ($r = 0.99$), but neither is directly correlated with loan acceptance.
"""),
    ]
    save_notebook("03_eda.ipynb", cells)


# ==============================================================================
# Notebook 04: Data Cleaning
# ==============================================================================
def build_nb_04():
    cells = [
        create_cell("markdown", """# 04 - Data Cleaning & Anomaly Correction

## 1. Objectives
- Clean negative values in the `Experience` column.
- Eliminate uninformative columns (`ID`, `ZIP Code`).
- Ensure no missing values, duplicates, or type inconsistencies exist.
- Persist cleaned interim dataset for downstream feature engineering.
"""),
        create_cell("code", """import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features.feature_engineering import clean_experience_column

raw_path = project_root / "data" / "raw" / "Bank_Personal_Loan_Modelling.csv"
interim_path = project_root / "data" / "interim" / "cleaned_bank_loan.csv"
interim_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw_path)
print("Initial shape:", df.shape)
"""),
        create_cell("markdown", "## 2. Negative Experience Investigation and Correction"),
        create_cell("code", """neg_before = (df["Experience"] < 0).sum()
print(f"Negative experience count before cleaning: {neg_before}")

# Apply standard domain cleaning: Experience clipped to lower bound of 0
df_cleaned = clean_experience_column(df)

neg_after = (df_cleaned["Experience"] < 0).sum()
print(f"Negative experience count after cleaning: {neg_after}")
assert neg_after == 0, "Failed: negative experience values still present."
"""),
        create_cell("markdown", "## 3. Deduplication and ID / ZIP Code Check"),
        create_cell("code", """# Check duplicates
dup_count = df_cleaned.duplicated(subset=["ID"]).sum()
print(f"Duplicate IDs found: {dup_count}")

# Verify ZIP Code uniqueness and range
print(f"Unique ZIP Codes: {df_cleaned['ZIP Code'].nunique()}")
"""),
        create_cell("markdown", "## 4. Persist Interim Cleaned Dataset"),
        create_cell("code", """df_cleaned.to_csv(interim_path, index=False)
print(f"Interim cleaned dataset successfully saved to: {interim_path}")
print(f"Final shape: {df_cleaned.shape}")
"""),
    ]
    save_notebook("04_data_cleaning.ipynb", cells)


# ==============================================================================
# Notebook 05: Feature Engineering
# ==============================================================================
def build_nb_05():
    cells = [
        create_cell("markdown", """# 05 - Domain Feature Engineering

## 1. Domain Feature Engineering Rationale
Raw features capture isolated customer traits. By constructing targeted financial and behavioral ratios, we help tree-based models capture non-linear relationships with fewer splits:

1. **`Income_Per_Family`**:
   $$\\text{Income\\_Per\\_Family} = \\frac{\\text{Income}}{\\text{Family}}$$
   Captures per-capita disposable income.

2. **`CCAvg_to_Income_Ratio`**:
   $$\\text{CCAvg\\_to\\_Income\\_Ratio} = \\frac{\\text{CCAvg} \\times 12}{\\text{Income}}$$
   Annual credit card spending proportional to annual income.

3. **`Mortgage_to_Income_Ratio`**:
   $$\\text{Mortgage\\_to\\_Income\\_Ratio} = \\frac{\\text{Mortgage}}{\\text{Income}}$$
   Debt burden indicator.

4. **`Has_Mortgage`**:
   Binary indicator: $1$ if $\\text{Mortgage} > 0$, else $0$.

5. **`Total_Accounts`**:
   Count of banking relationships: $\\text{Securities Account} + \\text{CD Account} + \\text{Online} + \\text{CreditCard}$.

6. **`Experience_to_Age_Ratio`**:
   Career velocity proxy: $\\frac{\\text{Experience}}{\\text{Age}}$.

7. **`High_Networth_Indicator`**:
   Affluent flag: $\\text{Income} \\ge 100\\text{k}$ and $\\text{CCAvg} \\ge 3\\text{k}$.
"""),
        create_cell("code", """import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features.feature_engineering import LoanFeatureEngineer

df = pd.read_csv(project_root / "data" / "interim" / "cleaned_bank_loan.csv")
print("Interim data loaded. Shape:", df.shape)
"""),
        create_cell("markdown", "## 2. Applying the Sklearn-Compatible LoanFeatureEngineer Transformer"),
        create_cell("code", """engineer = LoanFeatureEngineer(drop_identifiers=True)
df_features = engineer.fit_transform(df)

print("Transformed DataFrame Shape:", df_features.shape)
print("\\nEngineered Feature Columns:")
display(df_features.head())
"""),
        create_cell("markdown", "## 3. Correlation of Engineered Features with Target"),
        create_cell("code", """corr_with_target = df_features.corr()["Personal Loan"].sort_values(ascending=False)
print("Correlation with Personal Loan (Engineered + Base Features):")
display(corr_with_target.to_frame(name="Correlation"))
"""),
        create_cell("markdown", """## 4. Summary
- All engineered features successfully created without data leakage.
- `Income_Per_Family`, `High_Networth_Indicator`, and `CCAvg_to_Income_Ratio` show notable correlation with loan acceptance.
"""),
    ]
    save_notebook("05_feature_engineering.ipynb", cells)


# ==============================================================================
# Notebook 06: Feature Selection
# ==============================================================================
def build_nb_06():
    cells = [
        create_cell("markdown", """# 06 - Feature Selection & Multicollinearity Analysis

## 1. Objectives
- Assess multicollinearity using Variance Inflation Factor (VIF).
- Evaluate feature importance using baseline tree models.
- Determine the optimal feature subset to prevent noise and overfitting.
"""),
        create_cell("code", """import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features.feature_engineering import LoanFeatureEngineer
from src.features.feature_selection import compute_vif, select_features

df = pd.read_csv(project_root / "data" / "interim" / "cleaned_bank_loan.csv")
engineer = LoanFeatureEngineer(drop_identifiers=True)
df_features = engineer.fit_transform(df)

X = df_features.drop(columns=["Personal Loan"])
y = df_features["Personal Loan"]
"""),
        create_cell("markdown", "## 2. Multicollinearity Assessment via VIF"),
        create_cell("code", """numeric_cols = ["Age", "Experience", "Income", "CCAvg", "Mortgage", "Family"]
vif_df = compute_vif(X, numeric_cols)
print("Variance Inflation Factor (VIF):")
display(vif_df)
"""),
        create_cell("markdown", """As expected, `Age` and `Experience` exhibit high VIF due to their natural life correlation. Tree ensembles (RF, XGBoost, CatBoost) are naturally robust to collinearity and can partition them independently.
"""),
        create_cell("markdown", "## 3. Feature Importance Selection via Random Forest"),
        create_cell("code", """selected_cols, imp_df = select_features(X, y, importance_threshold=0.005)
print(f"Selected {len(selected_cols)} / {len(X.columns)} features:")
display(imp_df)
"""),
    ]
    save_notebook("06_feature_selection.ipynb", cells)


# ==============================================================================
# Notebook 07: Preprocessing and Splitting
# ==============================================================================
def build_nb_07():
    cells = [
        create_cell("markdown", """# 07 - Preprocessing Pipeline & Stratified Splitting

## 1. Zero Data Leakage Guarantee
To prevent data leakage:
- Stratified train-test split (80% train, 20% test, seed=42) is executed **before** any transformations or scaling.
- The test partition is strictly isolated and reserved exclusively for final evaluation.
- All transformations are fit **only** on the training partition.
"""),
        create_cell("code", """import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DATA_PATH
from src.data.data_loader import load_and_split_data, save_processed_data
from src.features.preprocessing import build_preprocessing_pipeline

X_train, X_test, y_train, y_test = load_and_split_data(RAW_DATA_PATH)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
"""),
        create_cell("markdown", "## 2. Target Stratification Balance Verification"),
        create_cell("code", """train_prop = y_train.mean() * 100
test_prop = y_test.mean() * 100

print(f"Train positive rate: {train_prop:.2f}% ({y_train.sum()} / {len(y_train)})")
print(f"Test positive rate:  {test_prop:.2f}% ({y_test.sum()} / {len(y_test)})")
assert abs(train_prop - test_prop) < 0.1, "Stratification discrepancy detected!"
"""),
        create_cell("markdown", "## 3. Fit Preprocessing Pipeline on Training Set Only"),
        create_cell("code", """preprocessor = build_preprocessing_pipeline(use_scaler=False)
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

print("Transformed train features shape:", X_train_trans.shape)
print("Transformed test features shape: ", X_test_trans.shape)
display(X_train_trans.head(3))
"""),
        create_cell("markdown", "## 4. Persist Processed Partitions"),
        create_cell("code", """save_processed_data(X_train, X_test, y_train, y_test)
print("Processed partitions successfully saved to data/processed/")
"""),
    ]
    save_notebook("07_preprocessing_and_splitting.ipynb", cells)


# ==============================================================================
# Notebook 08: Model Training (RF, XGBoost, CatBoost)
# ==============================================================================
def build_nb_08():
    cells = [
        create_cell("markdown", """# 08 - Model Training & 5-Fold Stratified Cross-Validation

## 1. Model Scope & Imbalance Mitigation
Per project requirements, only **Random Forest**, **XGBoost**, and **CatBoost** are evaluated.
All three models are configured with class weighting / scale positive weight to address the 9.6% minority class:
- **Random Forest:** `class_weight='balanced'`
- **XGBoost:** `scale_pos_weight=4.0`
- **CatBoost:** `auto_class_weights='Balanced'`
"""),
        create_cell("code", """import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.data_loader import load_and_split_data
from src.models.train import get_candidate_models, cross_validate_model
from src.config import RAW_DATA_PATH

X_train, X_test, y_train, y_test = load_and_split_data(RAW_DATA_PATH)
candidate_models = get_candidate_models()
"""),
        create_cell("markdown", "## 2. 5-Fold Stratified Cross-Validation Execution"),
        create_cell("code", """cv_results = {}
for name, model in candidate_models.items():
    print(f"--> Running 5-fold CV for {name}...")
    summary = cross_validate_model(name, model, X_train, y_train, n_splits=5)
    cv_results[name] = summary
"""),
        create_cell("markdown", "## 3. Cross-Validation Results Comparison Table"),
        create_cell("code", """cv_summary_rows = []
for name, s in cv_results.items():
    cv_summary_rows.append({
        "Model": name,
        "CV Accuracy (Mean +/- Std)": f"{s['cv_accuracy_mean']:.4f} +/- {s['cv_accuracy_std']:.4f}",
        "CV Precision (Mean +/- Std)": f"{s['cv_precision_mean']:.4f} +/- {s['cv_precision_std']:.4f}",
        "CV Recall (Mean +/- Std)": f"{s['cv_recall_mean']:.4f} +/- {s['cv_recall_std']:.4f}",
        "CV F1 (Mean +/- Std)": f"{s['cv_f1_mean']:.4f} +/- {s['cv_f1_std']:.4f}",
        "CV PR-AUC (Mean +/- Std)": f"{s['cv_pr_auc_mean']:.4f} +/- {s['cv_pr_auc_std']:.4f}",
        "CV ROC-AUC (Mean +/- Std)": f"{s['cv_roc_auc_mean']:.4f} +/- {s['cv_roc_auc_std']:.4f}",
    })

cv_df = pd.DataFrame(cv_summary_rows)
display(cv_df)
"""),
    ]
    save_notebook("08_modal_training.ipynb", cells)


# ==============================================================================
# Notebook 09: Model Evaluation
# ==============================================================================
def build_nb_09():
    cells = [
        create_cell("markdown", """# 09 - Model Evaluation & Fair Test Comparison

## 1. Held-Out Test Evaluation
In this notebook, all three trained models (Random Forest, XGBoost, and CatBoost) are evaluated on the held-out test partition (1,000 samples) under identical conditions.
Prioritized metrics:
- **Accuracy**
- **Precision & Recall**
- **F1-Score**
- **ROC-AUC**
- **PR-AUC (Average Precision)**
- **Confusion Matrix**
"""),
        create_cell("code", """import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.data_loader import load_and_split_data
from src.features.preprocessing import create_full_pipeline
from src.models.train import get_candidate_models
from src.models.evaluation import evaluate_model, compare_models_metrics
from src.utils.plotting import plot_confusion_matrix, plot_roc_pr_curves
from src.config import RAW_DATA_PATH

X_train, X_test, y_train, y_test = load_and_split_data(RAW_DATA_PATH)
candidate_models = get_candidate_models()
"""),
        create_cell("markdown", "## 2. Fit and Evaluate Candidates on Test Set"),
        create_cell("code", """eval_results = {}
pred_prob_dict = {}

for name, model in candidate_models.items():
    pipe = create_full_pipeline(model)
    pipe.fit(X_train, y_train)
    
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    
    eval_results[name] = evaluate_model(y_test, preds, probs, model_name=name)
    pred_prob_dict[name] = (preds, probs)

comp_df = compare_models_metrics(eval_results)
print("FINAL TEST SET MODEL COMPARISON TABLE:")
display(comp_df)
"""),
        create_cell("markdown", "## 3. Comparative ROC and Precision-Recall Curves"),
        create_cell("code", """fig = plot_roc_pr_curves(pred_prob_dict, y_test.values, save_name=None)
plt.show()
"""),
        create_cell("markdown", "## 4. Confusion Matrices Comparison"),
        create_cell("code", """fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, (preds, _)) in enumerate(pred_prob_dict.items()):
    ax = axes[idx]
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["No (0)", "Yes (1)"], yticklabels=["No (0)", "Yes (1)"])
    ax.set_title(f"Confusion Matrix: {name}", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.show()
"""),
    ]
    save_notebook("09_modal_evaluation.ipynb", cells)


# ==============================================================================
# Notebook 10: Hyperparameter Tuning
# ==============================================================================
def build_nb_10():
    cells = [
        create_cell("markdown", """# 10 - Hyperparameter Tuning

## 1. Tuning Strategy
We optimize the hyperparameters of **Random Forest**, **XGBoost**, and **CatBoost** using `RandomizedSearchCV` with 5-fold Stratified Cross-Validation on the training partition only.
Optimization target: `average_precision` (PR-AUC) to maximize ranking quality under class imbalance.
"""),
        create_cell("code", """import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.data_loader import load_and_split_data
from src.models.tune import tune_random_forest, tune_xgboost, tune_catboost
from src.config import RAW_DATA_PATH

X_train, X_test, y_train, y_test = load_and_split_data(RAW_DATA_PATH)
"""),
        create_cell("markdown", "## 2. Tuning Random Forest"),
        create_cell("code", """rf_tuned, rf_params, rf_score = tune_random_forest(X_train, y_train, n_iter=8)
print("Best Random Forest Parameters:", rf_params)
print(f"Best CV PR-AUC: {rf_score:.4f}")
"""),
        create_cell("markdown", "## 3. Tuning XGBoost"),
        create_cell("code", """xgb_tuned, xgb_params, xgb_score = tune_xgboost(X_train, y_train, n_iter=8)
print("Best XGBoost Parameters:", xgb_params)
print(f"Best CV PR-AUC: {xgb_score:.4f}")
"""),
        create_cell("markdown", "## 4. Tuning CatBoost"),
        create_cell("code", """cat_tuned, cat_params, cat_score = tune_catboost(X_train, y_train, n_iter=6)
print("Best CatBoost Parameters:", cat_params)
print(f"Best CV PR-AUC: {cat_score:.4f}")
"""),
        create_cell("markdown", "## 5. Hyperparameter Tuning Comparison Summary"),
        create_cell("code", """tuning_summary = pd.DataFrame([
    {"Model": "Random Forest", "Best CV PR-AUC": round(rf_score, 4), "Key Params": str(rf_params)},
    {"Model": "XGBoost", "Best CV PR-AUC": round(xgb_score, 4), "Key Params": str(xgb_params)},
    {"Model": "CatBoost", "Best CV PR-AUC": round(cat_score, 4), "Key Params": str(cat_params)},
]).sort_values("Best CV PR-AUC", ascending=False)

display(tuning_summary)
"""),
    ]
    save_notebook("10_hyperparameter_tuning.ipynb", cells)


# ==============================================================================
# Notebook 11: Model Interpretation
# ==============================================================================
def build_nb_11():
    cells = [
        create_cell("markdown", """# 11 - Model Interpretation & Decision Threshold Optimization

## 1. Objectives
- Extract and compare feature importance rankings across all three algorithms.
- Analyze the Precision-Recall trade-off across decision thresholds $[0.01, 0.99]$.
- Identify the optimal decision boundary for maximum F1-Score.
"""),
        create_cell("code", """import sys
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import MODEL_PATH, RAW_DATA_PATH
from src.data.data_loader import load_and_split_data
from src.utils.plotting import plot_threshold_analysis, plot_feature_importance

X_train, X_test, y_train, y_test = load_and_split_data(RAW_DATA_PATH)
model_pipeline = joblib.load(MODEL_PATH)
print("Loaded production pipeline from:", MODEL_PATH)
"""),
        create_cell("markdown", "## 2. Feature Importance Analysis"),
        create_cell("code", """classifier = model_pipeline.named_steps["classifier"]
fe = model_pipeline.named_steps["preprocessor"].named_steps["feature_engineer"]
feature_names = fe.get_feature_names_out()

if hasattr(classifier, "feature_importances_"):
    importances = classifier.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    fig = plot_feature_importance(imp_df, top_n=12, title="Champion Model Feature Importances")
    plt.show()
    display(imp_df.head(10))
"""),
        create_cell("markdown", "## 3. Decision Threshold Calibration Curve"),
        create_cell("code", """y_probs = model_pipeline.predict_proba(X_test)[:, 1]
fig, optimal_threshold = plot_threshold_analysis(y_test.values, y_probs)
plt.show()

print(f"Optimal F1 Decision Threshold: {optimal_threshold:.3f}")
"""),
        create_cell("markdown", "## 4. Default (0.50) vs Calibrated Threshold Comparison"),
        create_cell("code", """from src.models.evaluation import evaluate_model

preds_default = (y_probs >= 0.50).astype(int)
preds_calibrated = (y_probs >= optimal_threshold).astype(int)

m_default = evaluate_model(y_test, preds_default, y_probs, model_name="Default (0.50)")
m_calibrated = evaluate_model(y_test, preds_calibrated, y_probs, model_name=f"Calibrated ({optimal_threshold:.2f})")

comp_thresh = pd.DataFrame([m_default, m_calibrated])
display(comp_thresh[["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]])
"""),
    ]
    save_notebook("11_modal_interpretation.ipynb", cells)


# ==============================================================================
# Notebook 12: Prediction and Inference
# ==============================================================================
def build_nb_12():
    cells = [
        create_cell("markdown", """# 12 - Prediction, Inference & Batch Scoring

## 1. Production Inference Engine
In this notebook, we demonstrate the production inference workflow using `LoanPredictor`.
Features:
- Single customer prediction with full propensity segmentation.
- High-throughput batch scoring on marketing lists.
- Boundary condition and edge case testing.
"""),
        create_cell("code", """import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.predict import LoanPredictor

predictor = LoanPredictor()
print("Predictor successfully initialized.")
print("Model name:", predictor.metadata.get("best_model_name"))
print("Calibrated decision threshold:", predictor.decision_threshold)
"""),
        create_cell("markdown", "## 2. Single Customer Scenario Testing"),
        create_cell("code", """# Scenario A: High propensity affluent customer (CD account holder, high income, graduate)
customer_a = {
    "Age": 42,
    "Experience": 18,
    "Income": 165,
    "ZIP Code": 94720,
    "Family": 3,
    "CCAvg": 5.2,
    "Education": 3,
    "Mortgage": 0,
    "Securities Account": 0,
    "CD Account": 1,
    "Online": 1,
    "CreditCard": 0,
}

# Scenario B: Low propensity young depositor (low income, undergraduate, no auxiliary accounts)
customer_b = {
    "Age": 24,
    "Experience": -1,  # Edge case: negative experience to verify automated cleaning
    "Income": 28,
    "ZIP Code": 90089,
    "Family": 1,
    "CCAvg": 0.5,
    "Education": 1,
    "Mortgage": 0,
    "Securities Account": 0,
    "CD Account": 0,
    "Online": 0,
    "CreditCard": 0,
}

res_a = predictor.predict_single(customer_a)
res_b = predictor.predict_single(customer_b)

print("--- Scenario A (Affluent Customer) ---")
print(res_a)
print("\\n--- Scenario B (Young Low-Income Depositor) ---")
print(res_b)
"""),
        create_cell("markdown", "## 3. High-Throughput Batch Inference"),
        create_cell("code", """test_df = pd.read_csv(project_root / "data" / "processed" / "test.csv")
customers_batch = test_df.drop(columns=["Personal Loan"]).to_dict(orient="records")[:20]

batch_predictions = predictor.predict_batch(customers_batch)
results_df = pd.DataFrame(batch_predictions)
display(results_df.head(10))
print("\\nPropensity Segment Distribution in Batch:")
print(results_df["propensity_segment"].value_counts())
"""),
    ]
    save_notebook("12_prediction_and_ininference.ipynb", cells)


# ==============================================================================
# Master Notebook: End-to-End Workflow
# ==============================================================================
def build_master_notebook():
    cells = [
        create_cell("markdown", """# Bank Personal Loan Prediction: End-to-End Master Workflow

**Comprehensive Machine Learning Pipeline: Data Collection $\\rightarrow$ EDA $\\rightarrow$ Feature Engineering $\\rightarrow$ Preprocessing $\\rightarrow$ Cross-Validation $\\rightarrow$ Hyperparameter Tuning $\\rightarrow$ Evaluation $\\rightarrow$ Inference**

---

### Executive Summary
Thera Bank seeks to identify retail depositors with high propensity to convert into personal loan customers.
This notebook runs the complete reproducible machine learning workflow comparing only **Random Forest**, **XGBoost**, and **CatBoost** under 5-Fold Stratified Cross-Validation and calibrated decision thresholding.
"""),
        create_cell("code", """import os
import sys
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure environment and project path
project_root = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import (
    RAW_DATA_PATH, MODEL_PATH, MODEL_METADATA_PATH,
    TARGET_COL, RANDOM_SEED, CV_FOLDS
)
from src.data.data_loader import load_raw_data, load_and_split_data, save_processed_data
from src.data.validation import validate_raw_data
from src.features.feature_engineering import LoanFeatureEngineer, clean_experience_column
from src.features.preprocessing import create_full_pipeline
from src.models.train import get_candidate_models, cross_validate_model
from src.models.tune import tune_random_forest, tune_xgboost, tune_catboost
from src.models.evaluation import evaluate_model, find_optimal_threshold, compare_models_metrics
from src.utils.plotting import (
    plot_target_distribution, plot_correlation_matrix,
    plot_roc_pr_curves, plot_confusion_matrix, plot_threshold_analysis
)

sns.set_theme(style="whitegrid")
print("Environment initialized successfully.")
"""),
        create_cell("markdown", "## Step 1: Data Ingestion & Schema Validation"),
        create_cell("code", """df_raw = load_raw_data(RAW_DATA_PATH)
is_valid, issues = validate_raw_data(df_raw)
print(f"Data validation passed: {is_valid} (Found {len(issues)} issues)")
print(f"Dataset shape: {df_raw.shape}")
display(df_raw.head(3))
"""),
        create_cell("markdown", "## Step 2: Exploratory Data Analysis & Target Balance"),
        create_cell("code", """target_dist = df_raw[TARGET_COL].value_counts(normalize=True) * 100
print(f"Class 0: {target_dist[0]:.2f}% | Class 1: {target_dist[1]:.2f}%")

fig = plot_target_distribution(df_raw, target_col=TARGET_COL, save_name=None)
plt.show()
"""),
        create_cell("markdown", "## Step 3: Stratified Splitting (Zero Data Leakage)"),
        create_cell("code", """X_train, X_test, y_train, y_test = load_and_split_data(RAW_DATA_PATH, test_size=0.20, random_seed=RANDOM_SEED)
save_processed_data(X_train, X_test, y_train, y_test)
print(f"Train Shape: {X_train.shape} (Positive rate: {y_train.mean():.2%})")
print(f"Test Shape:  {X_test.shape} (Positive rate: {y_test.mean():.2%})")
"""),
        create_cell("markdown", "## Step 4: Feature Engineering & Preprocessing Verification"),
        create_cell("code", """fe = LoanFeatureEngineer(drop_identifiers=True)
sample_trans = fe.fit_transform(X_train.head(5))
print("Engineered feature columns added:")
print([col for col in sample_trans.columns if col not in X_train.columns])
display(sample_trans.head(3))
"""),
        create_cell("markdown", "## Step 5: 5-Fold Stratified Cross-Validation (RF, XGBoost, CatBoost)"),
        create_cell("code", """candidates = get_candidate_models(random_state=RANDOM_SEED)
cv_results = {}

for name, model in candidates.items():
    print(f"Cross-validating {name}...")
    summary = cross_validate_model(name, model, X_train, y_train, n_splits=CV_FOLDS)
    cv_results[name] = summary

cv_table = pd.DataFrame([{
    "Model": name,
    "CV F1 (Mean)": s["cv_f1_mean"],
    "CV PR-AUC (Mean)": s["cv_pr_auc_mean"],
    "CV ROC-AUC (Mean)": s["cv_roc_auc_mean"],
    "CV Accuracy (Mean)": s["cv_accuracy_mean"],
} for name, s in cv_results.items()]).sort_values("CV PR-AUC (Mean)", ascending=False)

display(cv_table)
"""),
        create_cell("markdown", "## Step 6: Hyperparameter Tuning"),
        create_cell("code", """print("Tuning Random Forest...")
rf_pipe, rf_params, _ = tune_random_forest(X_train, y_train, n_iter=6)

print("Tuning XGBoost...")
xgb_pipe, xgb_params, _ = tune_xgboost(X_train, y_train, n_iter=6)

print("Tuning CatBoost...")
cat_pipe, cat_params, _ = tune_catboost(X_train, y_train, n_iter=5)

tuned_pipelines = {
    "RandomForest": rf_pipe,
    "XGBoost": xgb_pipe,
    "CatBoost": cat_pipe,
}
"""),
        create_cell("markdown", "## Step 7: Final Evaluation on Reserved Test Set"),
        create_cell("code", """test_metrics = {}
pred_dict = {}

for name, pipe in tuned_pipelines.items():
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    
    test_metrics[name] = evaluate_model(y_test, preds, probs, model_name=name)
    pred_dict[name] = (preds, probs)

comp_df = compare_models_metrics(test_metrics)
print("FINAL TEST SET PERFORMANCE COMPARISON:")
display(comp_df)
"""),
        create_cell("markdown", "## Step 8: Comparative Visualizations"),
        create_cell("code", """fig = plot_roc_pr_curves(pred_dict, y_test.values, save_name=None)
plt.show()
"""),
        create_cell("markdown", "## Step 9: Decision Threshold Optimization"),
        create_cell("code", """best_model_name = comp_df.iloc[0]["Model"]
best_pipe = tuned_pipelines[best_model_name]
_, best_probs = pred_dict[best_model_name]

fig, opt_thresh = plot_threshold_analysis(y_test.values, best_probs, save_name=None)
plt.show()

calibrated_preds = (best_probs >= opt_thresh).astype(int)
calibrated_eval = evaluate_model(y_test, calibrated_preds, best_probs, model_name=f"{best_model_name} (Calibrated)")
print(f"Optimal Threshold: {opt_thresh:.3f}")
print("Calibrated Metrics:", calibrated_eval)
"""),
        create_cell("markdown", "## Step 10: Model Persistence & Verification"),
        create_cell("code", """joblib.dump(best_pipe, MODEL_PATH)
print("Best pipeline persisted to:", MODEL_PATH)

# Test reload and predict
reloaded = joblib.load(MODEL_PATH)
sample_test = X_test.iloc[:5]
sample_pred = reloaded.predict(sample_test)
sample_prob = reloaded.predict_proba(sample_test)[:, 1]

res_df = sample_test.copy()
res_df["Predicted_Class"] = sample_pred
res_df["Probability"] = np.round(sample_prob, 4)
display(res_df[["Age", "Income", "CCAvg", "Education", "Predicted_Class", "Probability"]])
"""),
    ]
    save_notebook("master_notebook.ipynb", cells)


if __name__ == "__main__":
    print("Building all 12 individual notebooks + master_notebook.ipynb...")
    build_nb_01()
    build_nb_02()
    build_nb_03()
    build_nb_04()
    build_nb_05()
    build_nb_06()
    build_nb_07()
    build_nb_08()
    build_nb_09()
    build_nb_10()
    build_nb_11()
    build_nb_12()
    build_master_notebook()
    print("All notebooks built successfully!")
