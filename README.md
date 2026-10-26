# Bank Personal Loan Prediction System

A production-grade, end-to-end Machine Learning system predicting customer acceptance propensity for Personal Loan marketing campaigns at Thera Bank. Built with clean modular architecture, leak-free preprocessing pipelines, 5-fold Stratified Cross-Validation, and hyperparameter-tuned ensemble models (**Random Forest**, **XGBoost**, and **CatBoost**).

---

## 1. Business Problem & Objective

**Thera Bank** has a large base of retail liability depositors. Converting liability depositors into personal loan (asset) customers increases net interest margins while retaining their primary deposit relationship. In a past marketing campaign, the baseline conversion rate was only **9.6%** (imbalanced target: 480 converters vs. 4,520 non-converters out of 5,000 customers).

The business goal is to:
1. Accurately identify depositors with high conversion likelihood.
2. Minimize marketing campaign acquisition costs and customer fatigue.
3. Prioritize metrics suited for severe class imbalance (**Recall**, **Precision**, **F1-Score**, **ROC-AUC**, and **PR-AUC**).
4. Provide a low-latency, production-ready inference API and reproducible notebooks.

---

## 2. Project Architecture & Directory Structure

```text
d:\Personal lone\
├── app.py                             # FastAPI production REST API (health, metadata, predict, batch)
├── train_pipeline.py                  # End-to-end CLI training & evaluation script
├── pyproject.toml                     # Project packaging and metadata
├── requirements.txt                   # Production environment dependencies
├── README.md                          # Comprehensive project documentation
├── .gitignore                         # Git exclusion rules
│
├── data/
│   ├── raw/
│   │   └── Bank_Personal_Loan_Modelling.csv  # Original immutable raw dataset (5,000 rows x 14 cols)
│   ├── interim/
│   │   └── cleaned_bank_loan.csv             # Cleaned interim data (experience anomaly corrected)
│   └── processed/
│       ├── train.csv                         # Stratified training partition (4,000 rows)
│       └── test.csv                          # Reserved held-out test partition (1,000 rows)
│
├── src/
│   ├── config.py                      # Global filesystem paths, schemas, seeds, and hyperparameter grids
│   ├── data/
│   │   ├── data_loader.py             # Ingestion, validation, and stratified splitting
│   │   └── validation.py              # Pydantic schemas and raw dataset integrity checks
│   ├── features/
│   │   ├── feature_engineering.py     # Domain ratios & sklearn-compatible LoanFeatureEngineer
│   │   ├── feature_selection.py       # VIF multicollinearity and tree importance selection
│   │   └── preprocessing.py           # Leak-free ColumnTransformer & Pipeline builder
│   ├── models/
│   │   ├── train.py                   # 5-fold Stratified CV for RF, XGBoost, and CatBoost
│   │   ├── tune.py                    # RandomizedSearchCV optimizing PR-AUC & F1
│   │   ├── evaluation.py              # Classification metrics, confusion matrix, threshold tuning
│   │   ├── evalution.py               # Compatibility alias for legacy naming
│   │   └── predict.py                 # LoanPredictor inference engine with propensity segmentation
│   └── utils/
│       ├── logging_config.py          # Unified console & file logger
│       └── plotting.py                # Visualizations (ROC/PR curves, confusion matrices, distributions)
│
├── notebook/
│   ├── 01_data_collection.ipynb               # Data collection & schema verification
│   ├── 02_data_understanding.ipynb            # Descriptive statistics & anomaly detection
│   ├── 03_eda.ipynb                           # Univariate & bivariate exploratory analysis
│   ├── 04_data_cleaning.ipynb                 # Negative experience resolution & column pruning
│   ├── 05_feature_engineering.ipynb           # Financial domain interaction ratios
│   ├── 06_feature_selection.ipynb             # Multicollinearity (VIF) & importance ranking
│   ├── 07_preprocessing_and_splitting.ipynb   # Stratified splitting (zero data leakage)
│   ├── 08_modal_training.ipynb                # 5-fold Stratified CV for RF, XGB, CatBoost
│   ├── 09_modal_evaluation.ipynb              # Fair model comparison on held-out test set
│   ├── 10_hyperparameter_tuning.ipynb         # Systematic hyperparameter search
│   ├── 11_modal_interpretation.ipynb          # Feature importance & threshold calibration
│   ├── 12_prediction_and_ininference.ipynb    # Model persistence & batch inference
│   └── master_notebook.ipynb                  # Fully sequential end-to-end master workflow
│
├── artifacts/
│   ├── models/
│   │   └── bank_personal_loan_model.joblib    # Best serialized complete pipeline
│   ├── modals/
│   │   └── bank_personal_lone_modal.joblib    # Mirror copy for compatibility
│   └── metadata/
│       └── model_metadata.json                # Champion parameters, threshold, and metrics
│
├── reports/
│   ├── figures/                       # Publication-quality charts (ROC, PR, confusion matrix)
│   ├── metrics/                       # JSON evaluation metric logs
│   └── logs/                          # Pipeline execution logs
│
└── tests/
    ├── test_data_loader.py            # Tests data loading, schema, and stratified split
    ├── test_preprocessing.py          # Tests cleaning, feature engineering, and pipeline
    └── test_prediction.py             # Tests single & batch inference and Pydantic validation
```

---

## 3. Data Dictionary & Anomaly Resolution

| Attribute | Type | Description | Cleaning & Transformation |
| :--- | :--- | :--- | :--- |
| **ID** | Integer | Customer identifier | Dropped (non-predictive) |
| **Age** | Integer | Customer age (23 – 67 years) | Retained as continuous feature |
| **Experience** | Integer | Professional experience (-3 – 43 years) | **52 negative values corrected** by clipping at 0 |
| **Income** | Float | Annual income in \$000 (8 – 224) | Dominant predictor ($r = 0.50$ with target) |
| **ZIP Code** | Integer | Residential ZIP Code | Dropped (high cardinality without spatial coordinates) |
| **Family** | Integer | Family size (1 – 4 members) | Retained; used in `Income_Per_Family` |
| **CCAvg** | Float | Monthly credit card spend in \$000 | Used in annual `CCAvg_to_Income_Ratio` |
| **Education** | Integer | 1: Undergrad, 2: Graduate, 3: Professional | Ordinal feature |
| **Mortgage** | Float | Value of house mortgage in \$000 | 68.8% zeros; used in `Has_Mortgage` indicator |
| **Securities Account**| Binary | Has securities account (0/1) | Auxiliary banking relationship feature |
| **CD Account** | Binary | Has Certificate of Deposit (0/1) | Highly predictive (>45% conversion for CD holders) |
| **Online** | Binary | Uses Internet banking (0/1) | Engagement channel feature |
| **CreditCard** | Binary | Has Universal Bank credit card (0/1) | Cross-sell indicator |
| **Personal Loan** | Binary | Accepted loan in last campaign (0/1) | **Target Variable** (90.4% Class 0 / 9.6% Class 1) |

### Domain Engineered Features:
1. **`Income_Per_Family`**: $\frac{\text{Income}}{\text{Family}}$ (household per-capita income).
2. **`CCAvg_to_Income_Ratio`**: $\frac{\text{CCAvg} \times 12}{\text{Income}}$ (annualized credit card expenditure burden).
3. **`Mortgage_to_Income_Ratio`**: $\frac{\text{Mortgage}}{\text{Income}}$ (debt-to-income ratio).
4. **`Has_Mortgage`**: Binary flag indicating active home mortgage.
5. **`Total_Accounts`**: Cumulative auxiliary banking relationship count ($\sum$ Securities, CD, Online, CreditCard).
6. **`Experience_to_Age_Ratio`**: Career velocity progression index.
7. **`High_Networth_Indicator`**: Binary flag ($\text{Income} \ge 100\text{k}$ and $\text{CCAvg} \ge 3\text{k}$).

---

## 4. Model Training, Validation & Leak-Free Pipeline

### Candidate Models
Per strict project specifications, only three ensemble architectures were evaluated:
1. **Random Forest Classifier**: Bagged decision trees with `class_weight='balanced'`.
2. **XGBoost Classifier**: Gradient boosted trees with `scale_pos_weight=4.0`.
3. **CatBoost Classifier**: Gradient boosted symmetric decision trees with `auto_class_weights='Balanced'`.

### Leak Prevention Protocol
- **Stratified Split (80/20):** 4,000 training observations and 1,000 test observations. Stratification guarantees exact $9.6\%$ positive balance in both splits.
- **Strict Partitioning:** The test set was held out and never accessed during feature engineering, VIF analysis, cross-validation, or hyperparameter tuning.
- **Sklearn Pipeline Encapsulation:** The custom transformer `LoanFeatureEngineer` is integrated directly into an `sklearn.pipeline.Pipeline`, ensuring all transformations operate cleanly on any new inference payload.

---

## 5. Actual Evaluation Results & Model Comparison

### 5-Fold Stratified Cross-Validation on Training Set (4,000 samples)
| Model | CV F1 (Mean $\pm$ Std) | CV PR-AUC (Mean $\pm$ Std) | CV ROC-AUC (Mean $\pm$ Std) | CV Accuracy (Mean $\pm$ Std) |
| :--- | :---: | :---: | :---: | :---: |
| **CatBoost** | **0.9276 $\pm$ 0.0256** | **0.9861 $\pm$ 0.0064** | **0.9983 $\pm$ 0.0008** | **98.65%** |
| **XGBoost** | 0.9117 $\pm$ 0.0169 | 0.9819 $\pm$ 0.0045 | 0.9978 $\pm$ 0.0006 | 98.32% |
| **RandomForest**| 0.8927 $\pm$ 0.0187 | 0.9724 $\pm$ 0.0052 | 0.9965 $\pm$ 0.0004 | 98.05% |

### Final Evaluation on Held-Out Test Set (1,000 samples)
*All metrics reflect genuine execution on the unseen test partition.*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (Champion)** | **99.20%** | **94.90%** | **96.88%** | **95.88%** | **0.9995** | **0.9955** | **99.45%** |
| **XGBoost** | **99.20%** | 94.90% | **96.88%** | **95.88%** | 0.9993 | 0.9944 | 99.45% |
| **RandomForest** | 99.00% | 93.88% | 95.83% | 0.9485 | 0.9989 | 0.9915 | 99.34% |

### Champion Model Selection & Threshold Calibration
- **Champion Selected:** `CatBoost` achieved the highest PR-AUC (**0.9955**) and ROC-AUC (**0.9995**).
- **Default Threshold (0.50):** Precision = 94.90%, Recall = 96.88%, F1 = 95.88%.
- **Optimal Calibrated Threshold (0.634):**
  - **Precision:** **96.88%** (+1.98% improvement)
  - **Recall:** **96.88%**
  - **F1-Score:** **96.88%** (+1.00% improvement)
  - True Positives: 93 / 96, False Positives: only 3 / 904.

### Top Predictive Feature Drivers (CatBoost Relative Importance):
1. **Income** ($24.90\%$)
2. **Education** ($20.45\%$)
3. **CCAvg** ($10.30\%$)
4. **Family** ($9.52\%$)
5. **Income_Per_Family** ($7.14\%$)
6. **CD Account** ($5.82\%$)

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.14 on Windows)
- Pip

### 1. Virtual Environment Setup & Dependencies
```powershell
# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies inside .venv
pip install -r requirements.txt
```

### 2. Run Automated Unit & Integration Tests
```bash
pytest tests
```
*Output: 10 passed in ~4.9s.*

### 3. Execute Full End-to-End Training Pipeline
```bash
python train_pipeline.py
```
*Runs data loading, stratified splitting, 5-fold CV, tuning for RF/XGB/CatBoost, test set evaluation, threshold calibration, figure generation, and saves the pipeline to `artifacts/models/bank_personal_loan_model.joblib`.*

### 4. Launch FastAPI REST Service
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation available at: **`http://localhost:8000/docs`**

#### Sample Inference Request (`POST /predict`):
```json
{
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
  "CreditCard": 0
}
```

#### Sample Response:
```json
{
  "prediction": 1,
  "loan_accepted": true,
  "acceptance_probability": 0.9984,
  "propensity_segment": "High Propensity",
  "decision_threshold_used": 0.634
}
```

### 5. Launch Jupyter Notebooks
```bash
jupyter lab
```
Navigate to `notebook/` to execute any of the 12 modular notebooks or `master_notebook.ipynb`.
