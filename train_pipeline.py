"""Main end-to-end training pipeline script.

Executes data loading, validation, stratified splitting, model training (RF, XGBoost, CatBoost),
cross-validation, hyperparameter tuning, evaluation, threshold calibration, and artifact persistence.

Usage:
    python train_pipeline.py
"""

import json
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    MODEL_PATH,
    MODEL_TYPO_PATH,
    MODEL_METADATA_PATH,
    METRICS_JSON_PATH,
    FIGURES_DIR,
    RANDOM_SEED,
    CV_FOLDS,
    TARGET_COL,
)
from src.data.data_loader import load_and_split_data, save_processed_data
from src.features.feature_selection import select_features
from src.models.train import get_candidate_models, cross_validate_model
from src.models.tune import tune_random_forest, tune_xgboost, tune_catboost
from src.models.evaluation import evaluate_model, find_optimal_threshold, compare_models_metrics
from src.features.preprocessing import create_full_pipeline
from src.utils.logging_config import get_logger
from src.utils.plotting import (
    plot_target_distribution,
    plot_correlation_matrix,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_feature_importance,
    plot_threshold_analysis,
)

logger = get_logger("train_pipeline")


def run_pipeline() -> None:
    """Executes the full machine learning training workflow."""
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING BANK PERSONAL LOAN PREDICTION TRAINING PIPELINE")
    logger.info("=" * 70)

    # 1. Load Data & Stratified Split (Ensures zero data leakage)
    logger.info("[Step 1/8] Loading dataset and performing stratified split...")
    X_train, X_test, y_train, y_test = load_and_split_data(RAW_DATA_PATH)
    save_processed_data(X_train, X_test, y_train, y_test)

    # Save target distribution plot
    raw_df = pd.concat([X_train, y_train], axis=1)
    plot_target_distribution(raw_df, target_col=TARGET_COL, save_name="target_distribution.png")
    plot_correlation_matrix(raw_df, save_name="correlation_matrix.png")

    # 2. Cross-Validation of Candidate Models (RF, XGBoost, CatBoost only)
    logger.info("[Step 2/8] Cross-validating Random Forest, XGBoost, and CatBoost (5-fold Stratified CV)...")
    candidate_models = get_candidate_models(random_state=RANDOM_SEED)
    cv_results = {}

    for name, model in candidate_models.items():
        summary = cross_validate_model(name, model, X_train, y_train, n_splits=CV_FOLDS)
        cv_results[name] = summary

    # 3. Hyperparameter Tuning
    logger.info("[Step 3/8] Running hyperparameter tuning on candidates...")
    tuned_models = {}

    rf_tuned, rf_params, _ = tune_random_forest(X_train, y_train, n_iter=10)
    tuned_models["RandomForest"] = (rf_tuned, rf_params)

    xgb_tuned, xgb_params, _ = tune_xgboost(X_train, y_train, n_iter=10)
    tuned_models["XGBoost"] = (xgb_tuned, xgb_params)

    cat_tuned, cat_params, _ = tune_catboost(X_train, y_train, n_iter=10)
    tuned_models["CatBoost"] = (cat_tuned, cat_params)

    # 4. Final Evaluation on Held-Out Test Partition
    logger.info("[Step 4/8] Evaluating tuned models on the reserved test partition...")
    test_eval_results = {}
    test_predictions_dict = {}

    for name, (pipeline, params) in tuned_models.items():
        # Pipeline is already fit by RandomizedSearchCV
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        eval_res = evaluate_model(y_test, preds, probs, model_name=name)
        test_eval_results[name] = eval_res
        test_predictions_dict[name] = (preds, probs)

        logger.info(
            "Test Metrics for %s: Acc=%.4f | Prec=%.4f | Rec=%.4f | F1=%.4f | ROC-AUC=%.4f | PR-AUC=%.4f",
            name,
            eval_res["accuracy"],
            eval_res["precision"],
            eval_res["recall"],
            eval_res["f1_score"],
            eval_res["roc_auc"],
            eval_res["pr_auc"],
        )

    # Print Comparison Table
    comparison_df = compare_models_metrics(test_eval_results)
    print("\n" + "=" * 80)
    print("FINAL TEST SET MODEL COMPARISON TABLE:")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80 + "\n")

    # 5. Model Selection (Prioritize PR-AUC and F1-score for imbalanced target)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline, best_params = tuned_models[best_model_name]
    best_preds, best_probs = test_predictions_dict[best_model_name]
    best_metrics = test_eval_results[best_model_name]

    logger.info("Champion Model Selected: %s", best_model_name)

    # 6. Decision Threshold Optimization for Champion Model
    logger.info("[Step 5/8] Optimizing decision threshold for champion model...")
    _, opt_threshold = plot_threshold_analysis(y_test.values, best_probs, save_name="threshold_analysis.png")
    thresh_preds = (best_probs >= opt_threshold).astype(int)
    threshold_metrics = evaluate_model(y_test, thresh_preds, best_probs, model_name=f"{best_model_name}_calibrated")
    logger.info(
        "Threshold Calibrated Metrics (Threshold=%.3f): Prec=%.4f | Rec=%.4f | F1=%.4f",
        opt_threshold,
        threshold_metrics["precision"],
        threshold_metrics["recall"],
        threshold_metrics["f1_score"],
    )

    # 7. Generate Evaluation Figures
    logger.info("[Step 6/8] Generating publication-quality figures...")
    plot_roc_pr_curves(test_predictions_dict, y_test.values, save_name="roc_pr_comparison.png")
    plot_confusion_matrix(y_test.values, best_preds, model_name=best_model_name, save_name="confusion_matrix.png")

    # Feature Importance
    classifier_step = best_pipeline.named_steps["classifier"]
    if hasattr(classifier_step, "feature_importances_"):
        fe_step = best_pipeline.named_steps["preprocessor"].named_steps["feature_engineer"]
        feature_names = fe_step.get_feature_names_out()
        importances = classifier_step.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        plot_feature_importance(imp_df, top_n=15, title=f"{best_model_name} Top Feature Importances", save_name="feature_importance.png")
        logger.info("Top 5 predictive features:\n%s", imp_df.head(5).to_string(index=False))

    # 8. Persist Artifacts
    logger.info("[Step 7/8] Persisting model pipeline and metadata...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info("Model pipeline saved to: %s", MODEL_PATH)

    # Mirror to typo directory for legacy compatibility
    MODEL_TYPO_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_TYPO_PATH)

    metadata = {
        "best_model_name": best_model_name,
        "hyperparameters": best_params,
        "optimal_threshold": opt_threshold,
        "test_metrics_default_threshold": best_metrics,
        "test_metrics_calibrated_threshold": threshold_metrics,
        "all_models_test_comparison": test_eval_results,
        "cv_summary": {k: {m: v for m, v in val.items() if m != "oof_probabilities"} for k, val in cv_results.items()},
        "dataset_shape": {
            "train": list(X_train.shape),
            "test": list(X_test.shape),
        },
        "target_balance": {
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "random_seed": RANDOM_SEED,
        "cv_folds": CV_FOLDS,
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved metadata to: %s", MODEL_METADATA_PATH)

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(test_eval_results, f, indent=4)
    logger.info("Saved metrics to: %s", METRICS_JSON_PATH)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY IN %.2f SECONDS", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    run_pipeline()
