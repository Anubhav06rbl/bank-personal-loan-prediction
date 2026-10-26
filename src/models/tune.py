"""Hyperparameter tuning module.

Optimizes Random Forest, XGBoost, and CatBoost models using Stratified K-Fold
Cross-Validation with focus on PR-AUC (Average Precision) and F1-Score.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.config import (
    RANDOM_SEED,
    CV_FOLDS,
    RF_PARAM_GRID,
    XGB_PARAM_GRID,
    CATBOOST_PARAM_GRID,
)
from src.features.preprocessing import create_full_pipeline
from src.utils.logging_config import get_logger

logger = get_logger("model_tune")


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 15,
    scoring: str = "average_precision",
    random_state: int = RANDOM_SEED,
) -> Tuple[Any, Dict[str, Any], float]:
    """Tunes RandomForestClassifier via RandomizedSearchCV."""
    logger.info("Tuning Random Forest (%d iterations, scoring=%s)...", n_iter, scoring)
    param_distributions = {f"classifier__{k}": v for k, v in RF_PARAM_GRID.items()}

    base_pipe = create_full_pipeline(RandomForestClassifier(random_state=random_state))
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    clean_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
    logger.info("Best RF Params: %s | Best Score (%s): %.4f", clean_params, scoring, search.best_score_)
    return search.best_estimator_, clean_params, float(search.best_score_)


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 15,
    scoring: str = "average_precision",
    random_state: int = RANDOM_SEED,
) -> Tuple[Any, Dict[str, Any], float]:
    """Tunes XGBClassifier via RandomizedSearchCV."""
    logger.info("Tuning XGBoost (%d iterations, scoring=%s)...", n_iter, scoring)
    param_distributions = {f"classifier__{k}": v for k, v in XGB_PARAM_GRID.items()}

    base_pipe = create_full_pipeline(
        XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1)
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    clean_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
    logger.info("Best XGBoost Params: %s | Best Score (%s): %.4f", clean_params, scoring, search.best_score_)
    return search.best_estimator_, clean_params, float(search.best_score_)


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 10,
    scoring: str = "average_precision",
    random_state: int = RANDOM_SEED,
) -> Tuple[Any, Dict[str, Any], float]:
    """Tunes CatBoostClassifier via RandomizedSearchCV."""
    logger.info("Tuning CatBoost (%d iterations, scoring=%s)...", n_iter, scoring)
    param_distributions = {f"classifier__{k}": v for k, v in CATBOOST_PARAM_GRID.items()}

    base_pipe = create_full_pipeline(
        CatBoostClassifier(random_seed=random_state, verbose=0, thread_count=-1)
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    clean_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
    logger.info("Best CatBoost Params: %s | Best Score (%s): %.4f", clean_params, scoring, search.best_score_)
    return search.best_estimator_, clean_params, float(search.best_score_)


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "average_precision",
) -> Dict[str, Dict[str, Any]]:
    """Runs hyperparameter tuning for all 3 supported models (RF, XGB, CatBoost)."""
    results = {}

    rf_pipe, rf_params, rf_score = tune_random_forest(X_train, y_train, scoring=scoring)
    results["RandomForest"] = {"pipeline": rf_pipe, "params": rf_params, "score": rf_score}

    xgb_pipe, xgb_params, xgb_score = tune_xgboost(X_train, y_train, scoring=scoring)
    results["XGBoost"] = {"pipeline": xgb_pipe, "params": xgb_params, "score": xgb_score}

    cat_pipe, cat_params, cat_score = tune_catboost(X_train, y_train, scoring=scoring)
    results["CatBoost"] = {"pipeline": cat_pipe, "params": cat_params, "score": cat_score}

    return results
