"""
shap_analysis.py
SHAP explainability for the selected model — this is the same explanation
machinery the agent's Risk Explanation Tool will call in Milestone P4, so
it's built as a reusable function now rather than a one-off script.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from ml.evaluation.metrics import FIGURES_DIR


def run_shap_summary(pipe, X_test, model_name: str, max_display: int = 10, save: bool = True):
    """Tree-based models use the exact, fast TreeExplainer. Linear models
    fall back to LinearExplainer. (KernelExplainer, which works on anything,
    is deliberately not the default here — it's far slower and unnecessary
    for the model families in this project.)"""
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]

    X_test_t = prep.transform(X_test)
    feature_names = prep.get_feature_names_out()
    X_test_t_df = pd.DataFrame(X_test_t, columns=feature_names)

    if hasattr(clf, "feature_importances_"):
        explainer = shap.TreeExplainer(clf)
    elif hasattr(clf, "coef_"):
        explainer = shap.LinearExplainer(clf, X_test_t_df)
    else:
        raise ValueError(f"No supported SHAP explainer for model type: {type(clf)}")

    shap_values = explainer.shap_values(X_test_t_df)

    # Normalize across shap-version return shapes to "positive class" values.
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif getattr(shap_values, "ndim", 2) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    fig = plt.figure()
    shap.summary_plot(sv, X_test_t_df, plot_type="bar", show=False, max_display=max_display)
    plt.title(f"SHAP feature importance — {model_name}")
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / f"shap_importance_{model_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    return sv, feature_names


def explain_single_prediction(pipe, patient_row: pd.DataFrame, model_name: str, top_n: int = 5):
    """Per-patient explanation — this is what the agent's Risk Explanation
    Tool will call for one patient at inference time (not the whole test
    set). Returns the top contributing features for THIS patient's
    prediction, signed (positive = pushes toward disease-present)."""
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]

    row_t = prep.transform(patient_row)
    feature_names = prep.get_feature_names_out()
    row_df = pd.DataFrame(row_t, columns=feature_names)

    explainer = shap.TreeExplainer(clf) if hasattr(clf, "feature_importances_") else shap.LinearExplainer(clf, row_df)
    shap_values = explainer.shap_values(row_df)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    elif getattr(shap_values, "ndim", 2) == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    contrib = (
        pd.DataFrame({"feature": feature_names, "shap_value": sv})
        .assign(abs_value=lambda d: d["shap_value"].abs())
        .sort_values("abs_value", ascending=False)
        .head(top_n)
        .drop(columns="abs_value")
    )
    return contrib
