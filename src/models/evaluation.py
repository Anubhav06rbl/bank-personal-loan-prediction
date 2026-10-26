"""Model evaluation module.

Computes comprehensive metrics tailored for imbalanced classification:
Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC (Average Precision),
Specificity, False Positive Rate, and optimal decision threshold tuning.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)

from src.utils.logging_config import get_logger

logger = get_logger("model_evaluation")


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Calculates all key evaluation metrics for binary classification.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted class labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (class 1).
        model_name: Identifier for the model.

    Returns:
        Dict[str, Any]: Comprehensive evaluation metrics dictionary.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))

        metrics["roc_auc"] = round(roc_auc, 4)
        metrics["pr_auc"] = round(pr_auc, 4)
        metrics["brier_score"] = round(brier, 4)

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Finds decision threshold that maximizes the specified metric (f1 or balanced).

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted positive class probabilities.
        metric: Metric to optimize ('f1' or 'precision_recall_balance').

    Returns:
        Tuple of (optimal_threshold, best_score).
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.50
    best_score = -1.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "precision_recall_balance":
            p = precision_score(y_true, preds, zero_division=0)
            r = recall_score(y_true, preds, zero_division=0)
            score = -abs(p - r)  # Closest to equality
        else:
            score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return round(best_thresh, 3), round(best_score, 4)


def compare_models_metrics(
    eval_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Formats a comparison dictionary of model evaluation outputs into a clean DataFrame."""
    records = []
    for model_name, m in eval_results.items():
        records.append(
            {
                "Model": model_name,
                "Accuracy": m.get("accuracy", np.nan),
                "Precision": m.get("precision", np.nan),
                "Recall": m.get("recall", np.nan),
                "F1-Score": m.get("f1_score", np.nan),
                "ROC-AUC": m.get("roc_auc", np.nan),
                "PR-AUC": m.get("pr_auc", np.nan),
                "Specificity": m.get("specificity", np.nan),
            }
        )
    return pd.DataFrame(records).sort_values(by="PR-AUC", ascending=False)
