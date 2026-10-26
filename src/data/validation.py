"""Data validation module.

Validates schemas, data types, missing values, and value ranges for raw datasets
and incoming inference payloads using Pydantic.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict

from src.config import FEATURE_COLS, TARGET_COL, FEATURE_RANGES, ID_COL
from src.utils.logging_config import get_logger

logger = get_logger("data_validation")


class CustomerLoanInput(BaseModel):
    """Pydantic schema for single-customer loan application input."""

    Age: int = Field(..., ge=18, le=100, description="Customer age in years")
    Experience: int = Field(..., ge=-5, le=80, description="Years of professional experience")
    Income: float = Field(..., ge=0, le=1000, description="Annual income ($000)")
    ZIP_Code: Optional[int] = Field(None, alias="ZIP Code", description="Customer ZIP Code")
    Family: int = Field(..., ge=1, le=10, description="Family size")
    CCAvg: float = Field(..., ge=0.0, le=50.0, description="Monthly credit card spend ($000)")
    Education: int = Field(..., ge=1, le=3, description="1: Undergrad, 2: Graduate, 3: Professional")
    Mortgage: float = Field(0.0, ge=0.0, le=2000.0, description="Value of house mortgage ($000)")
    Securities_Account: int = Field(0, alias="Securities Account", ge=0, le=1)
    CD_Account: int = Field(0, alias="CD Account", ge=0, le=1)
    Online: int = Field(1, ge=0, le=1, description="Internet banking user")
    CreditCard: int = Field(0, ge=0, le=1, description="Universal Bank credit card holder")

    model_config = ConfigDict(populate_by_name=True)


class BatchCustomerLoanInput(BaseModel):
    """Schema for batch inference requests."""

    customers: List[CustomerLoanInput]


def validate_raw_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validates raw dataset integrity before any processing.

    Args:
        df: Input raw DataFrame.

    Returns:
        Tuple of (is_valid: bool, issues: list of error descriptions).
    """
    issues = []

    # Check minimum row requirement
    if len(df) < 100:
        issues.append(f"Dataset too small: contains only {len(df)} rows.")

    # Check required columns
    expected_cols = [ID_COL, TARGET_COL] + FEATURE_COLS
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check missing values
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        issues.append(f"Columns with null values detected: {cols_with_nulls.to_dict()}")

    # Check target column values
    if TARGET_COL in df.columns:
        unique_targets = set(df[TARGET_COL].unique())
        if not unique_targets.issubset({0, 1}):
            issues.append(f"Target column '{TARGET_COL}' has unexpected values: {unique_targets}")

    # Check ranges
    for col, (min_val, max_val) in FEATURE_RANGES.items():
        if col in df.columns:
            actual_min = df[col].min()
            actual_max = df[col].max()
            if actual_min < min_val or actual_max > max_val:
                issues.append(
                    f"Column '{col}' out of bounds: expected [{min_val}, {max_val}], found [{actual_min}, {actual_max}]"
                )

    is_valid = len(issues) == 0
    if is_valid:
        logger.info("Raw data validation passed successfully.")
    else:
        logger.warning("Raw data validation encountered issues:\n%s", "\n".join(issues))

    return is_valid, issues


def validate_feature_schema(df: pd.DataFrame) -> bool:
    """Validates feature presence and basic data consistency for inference payloads."""
    required = [c for c in FEATURE_COLS if c not in ["ID", "ZIP Code"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Inference payload is missing required features: {missing}")
    return True
