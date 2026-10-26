"""Plotting and visualization utilities.

Generates high-resolution publication-quality plots for EDA, model metrics,
ROC / PR curves, confusion matrices, and feature importances.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless/script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from src.config import FIGURES_DIR

# Standard plot aesthetics
sns.set_theme(style="whitegrid", font_scale=1.0)
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "Personal Loan",
    save_name: Optional[str] = "target_distribution.png",
) -> plt.Figure:
    """Plots count and percentage distribution of the target class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100

    sns.barplot(x=counts.index, y=counts.values, ax=axes[0], palette=["#2b5c8f", "#d95f02"])
    axes[0].set_title(f"Target Count Distribution ({target_col})", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Class (0: Did Not Accept, 1: Accepted Loan)")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 40, f"{v:,}", ha="center", fontweight="bold")

    axes[1].pie(
        percentages,
        labels=[f"Class 0: {percentages[0]:.1f}%", f"Class 1: {percentages[1]:.1f}%"],
        autopct="%1.1f%%",
        colors=["#2b5c8f", "#d95f02"],
        startangle=90,
        explode=(0, 0.1),
    )
    axes[1].set_title("Target Proportion (Imbalance Ratio)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches="tight")
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    save_name: Optional[str] = "correlation_matrix.png",
) -> plt.Figure:
    """Plots a Spearman or Pearson correlation heatmap."""
    data_to_plot = df[columns] if columns else df.select_dtypes(include=[np.number])
    corr = data_to_plot.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)

    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Plots confusion matrix with annotations and normalized rates."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = np.array(
        [
            [f"{cm[0, 0]}\n({cm_norm[0, 0]:.1%})", f"{cm[0, 1]}\n({cm_norm[0, 1]:.1%})"],
            [f"{cm[1, 0]}\n({cm_norm[1, 0]:.1%})", f"{cm[1, 1]}\n({cm_norm[1, 1]:.1%})"],
        ]
    )

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Loan (0)", "Loan (1)"],
        yticklabels=["No Loan (0)", "Loan (1)"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")

    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches="tight")
    return fig


def plot_roc_pr_curves(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_test: np.ndarray,
    save_name: Optional[str] = "roc_pr_comparison.png",
) -> plt.Figure:
    """Plots comparative ROC and Precision-Recall curves across models.

    Args:
        models_dict: Dictionary mapping model names to (y_pred, y_prob).
        y_test: Ground truth labels.
        save_name: Optional filename for saving.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curves
    axes[0].plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)")
    for name, (_, y_prob) in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_score = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_score:.4f})")

    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate", fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Recall)", fontweight="bold")
    axes[0].set_title("ROC Curves Comparison", fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower right")

    # PR Curves
    baseline_pr = np.mean(y_test)
    axes[1].plot([0, 1], [baseline_pr, baseline_pr], "k--", label=f"Baseline ({baseline_pr:.3f})")
    for name, (_, y_prob) in models_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_score = average_precision_score(y_test, y_prob)
        axes[1].plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_score:.4f})")

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", fontweight="bold")
    axes[1].set_ylabel("Precision", fontweight="bold")
    axes[1].set_title("Precision-Recall Curves Comparison", fontsize=13, fontweight="bold")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches="tight")
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature Importances",
    save_name: Optional[str] = "feature_importance.png",
) -> plt.Figure:
    """Plots horizontal bar chart of top N important features."""
    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, int(top_n * 0.35))))
    bars = ax.barh(plot_df["feature"], plot_df["importance"], color="#2b5c8f", edgecolor="black")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score", fontweight="bold")
    ax.set_ylabel("Feature", fontweight="bold")

    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches="tight")
    return fig


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_name: Optional[str] = "threshold_analysis.png",
) -> Tuple[plt.Figure, float]:
    """Plots precision, recall, and F1-score across decision thresholds.

    Returns:
        (plt.Figure, optimal_threshold)
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    opt_idx = int(np.argmax(f1s))
    best_thresh = float(thresholds[opt_idx])
    best_f1 = f1s[opt_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#1f77b4", lw=2)
    ax.plot(thresholds, recalls, label="Recall", color="#2ca02c", lw=2)
    ax.plot(thresholds, f1s, label=f"F1-Score (Max: {best_f1:.4f} at {best_thresh:.2f})", color="#d62728", lw=2.5)

    ax.axvline(best_thresh, color="gray", linestyle="--", alpha=0.7, label=f"Optimal F1 Threshold ({best_thresh:.2f})")
    ax.axvline(0.5, color="black", linestyle=":", alpha=0.7, label="Default Threshold (0.50)")

    ax.set_title("Decision Threshold Tuning Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Probability Threshold", fontweight="bold")
    ax.set_ylabel("Metric Score", fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="lower center")

    plt.tight_layout()
    if save_name:
        fig.savefig(FIGURES_DIR / save_name, dpi=300, bbox_inches="tight")
    return fig, best_thresh
