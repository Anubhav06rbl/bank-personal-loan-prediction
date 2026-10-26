"""Feature engineering module.

Contains functions and an sklearn-compatible Transformer for domain feature
engineering, data anomaly correction (negative Experience values), and ratio creations.
"""

from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import DROPPED_COLS
from src.utils.logging_config import get_logger

logger = get_logger("feature_engineering")


def clean_experience_column(df: pd.DataFrame) -> pd.DataFrame:
    """Corrects negative experience values by clipping at zero.

    Negative experience values (typically -1, -2, -3) occur when customer age is low
    (e.g., 23-25) and represent zero or newly entered workforce experience.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame with cleaned Experience column.
    """
    df_clean = df.copy()
    if "Experience" in df_clean.columns:
        neg_count = (df_clean["Experience"] < 0).sum()
        if neg_count > 0:
            logger.info("Correcting %d negative values in 'Experience' column.", neg_count)
            df_clean["Experience"] = df_clean["Experience"].clip(lower=0)
    return df_clean


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for bank loan feature engineering.

    Creates financial domain features:
    - Experience anomaly correction
    - Income per family member
    - Annual credit card spending ratio to income
    - Mortgage to income ratio
    - Binary mortgage indicator
    - Cumulative banking engagement (total accounts)
    - Experience to age progression ratio
    - High net worth flag
    - Safely drops non-predictive identifiers (ID, ZIP Code)
    """

    def __init__(self, drop_identifiers: bool = True):
        self.drop_identifiers = drop_identifiers
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit method (stateless transformer)."""
        # Feature names will be determined on transform
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input features into engineered domain representations.

        Args:
            X: Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame with engineered features.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df = X.copy()

        # 1. Clean Experience anomaly
        if "Experience" in df.columns:
            df["Experience"] = df["Experience"].clip(lower=0)

        # 2. Ratio & Interaction Features
        # Safe division with small epsilon to prevent ZeroDivisionError
        eps = 1e-6

        if "Income" in df.columns and "Family" in df.columns:
            df["Income_Per_Family"] = df["Income"] / (df["Family"] + eps)

        if "CCAvg" in df.columns and "Income" in df.columns:
            # CCAvg is monthly ($000), Income is annual ($000)
            df["CCAvg_to_Income_Ratio"] = (df["CCAvg"] * 12) / (df["Income"] + eps)

        if "Mortgage" in df.columns and "Income" in df.columns:
            df["Mortgage_to_Income_Ratio"] = df["Mortgage"] / (df["Income"] + eps)
            df["Has_Mortgage"] = (df["Mortgage"] > 0).astype(int)

        # 3. Bank relationship depth (total auxiliary accounts used)
        account_cols = [c for c in ["Securities Account", "CD Account", "Online", "CreditCard"] if c in df.columns]
        if account_cols:
            df["Total_Accounts"] = df[account_cols].sum(axis=1)

        # 4. Career progression ratio
        if "Experience" in df.columns and "Age" in df.columns:
            df["Experience_to_Age_Ratio"] = df["Experience"] / np.maximum(df["Age"], 1)

        # 5. Affluent segment indicator
        if "Income" in df.columns and "CCAvg" in df.columns:
            df["High_Networth_Indicator"] = ((df["Income"] >= 100) & (df["CCAvg"] >= 3.0)).astype(int)

        # 6. Drop uninformative identifiers
        if self.drop_identifiers:
            cols_to_drop = [c for c in DROPPED_COLS if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

        self.feature_names_out_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns feature names after transformation."""
        return self.feature_names_out_
