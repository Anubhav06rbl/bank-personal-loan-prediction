"""Alias module for evaluation.py to support legacy spelling."""

from src.models.evaluation import (
    evaluate_model,
    find_optimal_threshold,
    compare_models_metrics,
)

__all__ = [
    "evaluate_model",
    "find_optimal_threshold",
    "compare_models_metrics",
]
