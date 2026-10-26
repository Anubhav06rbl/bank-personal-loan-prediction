"""Configuration module for the Bank Personal Loan Prediction project.

Defines all filesystem paths, dataset schemas, random states, model configurations,
and hyperparameter search spaces to ensure reproducibility across modules and notebooks.
"""

from pathlib import Path
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Base Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
MODALS_TYPO_DIR = ARTIFACTS_DIR / "modals"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
LOGS_DIR = REPORTS_DIR / "logs"

# Ensure essential directories exist
for path in [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    MODALS_TYPO_DIR,
    METADATA_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    LOGS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
RAW_DATA_PATH = RAW_DATA_DIR / "Bank_Personal_Loan_Modelling.csv"
CLEANED_DATA_PATH = INTERIM_DATA_DIR / "cleaned_bank_loan.csv"
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

MODEL_PATH = MODELS_DIR / "bank_personal_loan_model.joblib"
MODEL_TYPO_PATH = MODALS_TYPO_DIR / "bank_personal_lone_modal.joblib"
MODEL_METADATA_PATH = METADATA_DIR / "model_metadata.json"
METRICS_JSON_PATH = METRICS_DIR / "evaluation_metrics.json"

# ---------------------------------------------------------------------------
# Global Parameters & Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Data Schema & Feature Definitions
# ---------------------------------------------------------------------------
ID_COL = "ID"
TARGET_COL = "Personal Loan"

# Raw input features (excluding ID and Target)
FEATURE_COLS = [
    "Age",
    "Experience",
    "Income",
    "ZIP Code",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
    "Securities Account",
    "CD Account",
    "Online",
    "CreditCard",
]

NUMERICAL_COLS = [
    "Age",
    "Experience",
    "Income",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
]

BINARY_COLS = [
    "Securities Account",
    "CD Account",
    "Online",
    "CreditCard",
]

CATEGORICAL_COLS = [
    "Education",  # 1: Undergrad, 2: Graduate, 3: Advanced/Professional
]

DROPPED_COLS = [
    "ID",
    "ZIP Code",  # High-cardinality geographic identifier without regional encoding
]

# Engineered Feature Names
ENGINEERED_COLS = [
    "Income_Per_Family",
    "CCAvg_to_Income_Ratio",
    "Mortgage_to_Income_Ratio",
    "Has_Mortgage",
    "Total_Accounts",
    "Experience_to_Age_Ratio",
    "High_Networth_Indicator",
]

# Valid expected types and ranges for schema validation
FEATURE_RANGES = {
    "Age": (18, 100),
    "Experience": (-5, 80),  # Raw may contain negative values to be cleaned
    "Income": (0, 1000),
    "Family": (1, 10),
    "CCAvg": (0.0, 50.0),
    "Education": (1, 3),
    "Mortgage": (0, 2000),
    "Securities Account": (0, 1),
    "CD Account": (0, 1),
    "Online": (0, 1),
    "CreditCard": (0, 1),
}

# ---------------------------------------------------------------------------
# Baseline Hyperparameter Search Spaces for Tuning
# ---------------------------------------------------------------------------
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 6, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced", "balanced_subsample", None],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "scale_pos_weight": [1.0, 5.0, 9.0],
}

CATBOOST_PARAM_GRID = {
    "iterations": [150, 300, 500],
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.08, 0.15],
    "l2_leaf_reg": [1, 3, 5, 9],
}
