"""
metrics.py
Evaluation utilities for MedAgent's ML layer: confusion matrix, ROC curve
overlay across models, and feature importance / coefficient plots.

Kept separate from train.py so evaluation logic is independently testable
and reusable once the agent's Risk Explanation Tool needs the same plots.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display available in this environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

FIGURES_DIR = Path("outputs/figures")


def plot_confusion_matrix(y_test, y_pred, model_name: str, save: bool = True):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No disease", "Disease"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix — {model_name}")
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return cm


def plot_roc_curves(fitted_pipelines: dict, X_test, y_test, save: bool = True):
    """Overlay ROC curves for every model on one figure — makes the
    cv-vs-test comparison table visually obvious in the report."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pipe in fitted_pipelines.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC curves — all candidate models (test set)")
    ax.legend(loc="lower right", fontsize=8)
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "roc_curves_all_models.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(pipe, model_name: str, top_n: int = 10, save: bool = True):
    """Works for tree-based models (feature_importances_) and linear models
    (coef_) transparently — picks whichever attribute is present."""
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    feature_names = prep.get_feature_names_out()

    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        xlabel = "Importance (impurity-based)"
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
        xlabel = "|Coefficient| (standardized features)"
    else:
        raise ValueError(f"Model {model_name} exposes neither feature_importances_ nor coef_")

    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#3B6EA5")
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top {top_n} features — {model_name}")
    fig.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"feature_importance_{model_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return imp_df
