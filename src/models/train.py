"""Model training module.

Trains and evaluates only Random Forest, XGBoost, and CatBoost classifiers
using Stratified K-Fold Cross-Validation and class imbalance handling.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.config import RANDOM_SEED, CV_FOLDS
from src.features.preprocessing import create_full_pipeline
from src.models.evaluation import evaluate_model
from src.utils.logging_config import get_logger

logger = get_logger("model_train")


def get_candidate_models(random_state: int = RANDOM_SEED) -> Dict[str, Any]:
    """Instantiates candidate models: Random Forest, XGBoost, and CatBoost.

    Only these three models are permitted per project requirements.
    Class weights/scale_pos_weight are configured to address class imbalance (~9.6% positive).
    """
    # Imbalance ratio ~ 90.4 / 9.6 ~= 9.4
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=4.0,  # Balanced penalty for minority class false negatives
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=250,
            depth=6,
            learning_rate=0.08,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=random_state,
            thread_count=-1,
        ),
    }
    return models


def cross_validate_model(
    model_name: str,
    raw_model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Performs Stratified K-Fold Cross-Validation on a model pipeline.

    Evaluates out-of-fold performance to guard against overfitting and data leakage.

    Args:
        model_name: Name of the model.
        raw_model: Unfitted classifier.
        X_train: Training features.
        y_train: Training labels.
        n_splits: Number of cross-validation folds.
        random_state: Random state for reproducibility.

    Returns:
        Dict[str, Any]: Cross-validation metrics (fold scores, mean, std).
    """
    logger.info("Executing %d-fold Stratified CV for %s...", n_splits, model_name)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = []
    oof_predictions = np.zeros(len(y_train))
    oof_probabilities = np.zeros(len(y_train))

    y_train_arr = np.asarray(y_train)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_arr), 1):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train_arr[train_idx], y_train_arr[val_idx]

        # Construct leak-free pipeline for this fold
        fold_pipeline = create_full_pipeline(classifier=raw_model)
        fold_pipeline.fit(X_fold_train, y_fold_train)

        val_preds = fold_pipeline.predict(X_fold_val)
        val_probs = fold_pipeline.predict_proba(X_fold_val)[:, 1]

        oof_predictions[val_idx] = val_preds
        oof_probabilities[val_idx] = val_probs

        metrics = evaluate_model(y_fold_val, val_preds, val_probs, model_name=f"{model_name}_fold{fold}")
        fold_metrics.append(metrics)

    # Compute aggregate metrics across folds
    df_metrics = pd.DataFrame(fold_metrics)
    summary = {
        "model_name": model_name,
        "cv_accuracy_mean": float(df_metrics["accuracy"].mean()),
        "cv_accuracy_std": float(df_metrics["accuracy"].std()),
        "cv_precision_mean": float(df_metrics["precision"].mean()),
        "cv_precision_std": float(df_metrics["precision"].std()),
        "cv_recall_mean": float(df_metrics["recall"].mean()),
        "cv_recall_std": float(df_metrics["recall"].std()),
        "cv_f1_mean": float(df_metrics["f1_score"].mean()),
        "cv_f1_std": float(df_metrics["f1_score"].std()),
        "cv_roc_auc_mean": float(df_metrics["roc_auc"].mean()),
        "cv_roc_auc_std": float(df_metrics["roc_auc"].std()),
        "cv_pr_auc_mean": float(df_metrics["pr_auc"].mean()),
        "cv_pr_auc_std": float(df_metrics["pr_auc"].std()),
        "oof_metrics": evaluate_model(y_train_arr, oof_predictions, oof_probabilities, model_name=model_name),
        "oof_probabilities": oof_probabilities,
    }

    logger.info(
        "%s CV Results -> F1: %.4f (+/- %.4f) | PR-AUC: %.4f (+/- %.4f) | ROC-AUC: %.4f (+/- %.4f)",
        model_name,
        summary["cv_f1_mean"],
        summary["cv_f1_std"],
        summary["cv_pr_auc_mean"],
        summary["cv_pr_auc_std"],
        summary["cv_roc_auc_mean"],
        summary["cv_roc_auc_std"],
    )

    return summary


def train_and_compare_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Trains Random Forest, XGBoost, and CatBoost pipelines, compares CV & test performance.

    Returns:
        Tuple: (fitted_pipelines_dict, cv_summaries_dict, best_model_name)
    """
    models = get_candidate_models()
    fitted_pipelines = {}
    cv_summaries = {}

    for name, model in models.items():
        # Cross-validation on training partition only
        cv_summary = cross_validate_model(name, model, X_train, y_train)
        cv_summaries[name] = cv_summary

        # Fit complete pipeline on entire X_train
        pipeline = create_full_pipeline(classifier=model)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

    # Select best model based on validation PR-AUC mean (best metric for imbalanced data)
    best_model_name = max(cv_summaries.keys(), key=lambda k: cv_summaries[k]["cv_pr_auc_mean"])
    logger.info("Best candidate model determined by CV PR-AUC is: %s", best_model_name)

    return fitted_pipelines, cv_summaries, best_model_name
