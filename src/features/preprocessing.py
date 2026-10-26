"""Preprocessing and end-to-end pipeline creation module.

Assembles scikit-learn ColumnTransformers and complete Pipelines ensuring zero
data leakage between train and test distributions.
"""

from typing import List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.features.feature_engineering import LoanFeatureEngineer
from src.utils.logging_config import get_logger

logger = get_logger("preprocessing")


class DataFrameColumnTransformer(BaseEstimator, TransformerMixin):
    """Wraps ColumnTransformer to retain DataFrame column names and types."""

    def __init__(self, continuous_cols: Optional[List[str]] = None, scaler_type: str = "standard"):
        self.continuous_cols = continuous_cols or []
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.columns_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        cols_to_scale = [c for c in self.continuous_cols if c in X.columns]
        if cols_to_scale:
            self.scaler.fit(X[cols_to_scale])
        self.columns_out_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_out_)

        df = X.copy()
        cols_to_scale = [c for c in self.continuous_cols if c in df.columns]
        if cols_to_scale:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        return df


def build_preprocessing_pipeline(
    continuous_cols: Optional[List[str]] = None,
    use_scaler: bool = True,
) -> Pipeline:
    """Constructs a preprocessing pipeline containing feature engineering and scaling.

    Args:
        continuous_cols: List of continuous features to scale.
        use_scaler: If True, applies standard scaling to continuous columns.

    Returns:
        Pipeline: Preprocessing pipeline.
    """
    if continuous_cols is None:
        continuous_cols = [
            "Age",
            "Experience",
            "Income",
            "Family",
            "CCAvg",
            "Mortgage",
            "Income_Per_Family",
            "CCAvg_to_Income_Ratio",
            "Mortgage_to_Income_Ratio",
            "Experience_to_Age_Ratio",
        ]

    steps = [
        ("feature_engineer", LoanFeatureEngineer(drop_identifiers=True)),
    ]

    if use_scaler:
        steps.append(
            ("scaler", DataFrameColumnTransformer(continuous_cols=continuous_cols, scaler_type="standard"))
        )

    return Pipeline(steps)


def create_full_pipeline(
    classifier: BaseEstimator,
    use_scaler: bool = False,  # Tree ensembles (RF, XGB, CatBoost) are scale-invariant
) -> Pipeline:
    """Builds an end-to-end inference-ready Pipeline including feature engineering and estimator.

    Args:
        classifier: Trained or untrained classifier (RandomForest, XGBoost, CatBoost).
        use_scaler: Whether to include continuous feature standard scaling.

    Returns:
        Pipeline: Complete sklearn Pipeline.
    """
    preprocessor = build_preprocessing_pipeline(use_scaler=use_scaler)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline
