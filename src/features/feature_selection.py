"""Feature selection module.

Analyzes correlation, multicollinearity (VIF), and tree-based feature importance
to select optimal subsets of features for model training.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import TARGET_COL, RANDOM_SEED
from src.utils.logging_config import get_logger

logger = get_logger("feature_selection")


def compute_feature_correlations(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.Series:
    """Computes correlation of each numeric feature with the target column.

    Args:
        df: Input DataFrame with target included.
        target_col: Name of target column.

    Returns:
        pd.Series sorted by absolute correlation with target descending.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in numeric features.")

    corrs = numeric_df.corr()[target_col].drop(target_col)
    return corrs.reindex(corrs.abs().sort_values(ascending=False).index)


def compute_vif(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Calculates Variance Inflation Factor (VIF) to detect multicollinearity.

    Args:
        df: Input DataFrame.
        feature_cols: Numeric columns to test.

    Returns:
        pd.DataFrame: Features with corresponding VIF values.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df[feature_cols].dropna().astype(float)
    # Add constant column for intercept in regression
    X_with_const = X.assign(const=1.0)

    vif_data = []
    for i, col in enumerate(feature_cols):
        vif = variance_inflation_factor(X_with_const.values, i)
        vif_data.append({"feature": col, "VIF": vif})

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
    return vif_df


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: Optional[int] = None,
    importance_threshold: float = 0.005,
) -> Tuple[List[str], pd.DataFrame]:
    """Selects top features based on quick Random Forest feature importances.

    Args:
        X_train: Preprocessed/engineered training features.
        y_train: Training target labels.
        top_n: Maximum number of features to select (optional).
        importance_threshold: Minimum importance cutoff.

    Returns:
        Tuple of (selected_features_list, feature_importance_dataframe)
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, class_weight="balanced")
    rf.fit(X_train, y_train)

    importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    selected = importance_df[importance_df["importance"] >= importance_threshold]["feature"].tolist()
    if top_n is not None:
        selected = selected[:top_n]

    logger.info("Selected %d / %d features above importance threshold %.4f", len(selected), len(X_train.columns), importance_threshold)
    return selected, importance_df
