"""
train.py
Milestone P1 (model training + comparison) for MedAgent.

Trains and hyperparameter-tunes four classifiers (Logistic Regression,
Decision Tree, Random Forest, XGBoost) on the cleaned heart disease data,
compares them on held-out test data, and saves the selected model.

Design choices (see docs/architecture.md Section 5.6 for the full rationale):
- Primary tuning metric is ROC-AUC, not accuracy: for a clinical screening
  task, ranking quality across thresholds matters more than a single
  threshold's accuracy, and it's threshold-independent so it's a fairer
  basis for comparing models before we've picked an operating point.
- Model selection weighs recall alongside ROC-AUC: a false negative (telling
  a patient with heart disease they're low-risk) is more costly than a
  false positive in a screening context, so we don't just take the highest
  accuracy model.
- Categorical (nominal) columns are one-hot encoded for ALL models, not just
  linear ones. Tree models could technically split on the raw integer codes,
  but those codes are unordered categories (see preprocessing.py), and
  splitting on them as if ordered risks a subtly wrong (if not always
  visibly wrong) tree structure. Scaling is applied only for Logistic
  Regression, since tree-based models are invariant to monotonic feature
  scaling.
"""

import logging
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ml.training.preprocessing import (
    BINARY_COLUMNS,
    NOMINAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("ml/models")
RANDOM_STATE = 42


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """One preprocessor builder shared by every model, so the only
    difference between a linear-model pipeline and a tree-model pipeline is
    whether numeric features get scaled — everything else stays identical,
    which keeps the comparison fair."""
    num_transform = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL_COLUMNS),
            ("num", num_transform, NUMERIC_COLUMNS),
            ("bin", "passthrough", BINARY_COLUMNS),
        ]
    )


MODEL_CONFIGS = {
    "logistic_regression": dict(
        estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        scale_numeric=True,
        param_grid={"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]},
    ),
    "decision_tree": dict(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        scale_numeric=False,
        param_grid={
            "clf__max_depth": [3, 5, 7, None],
            "clf__min_samples_leaf": [1, 3, 5],
        },
    ),
    "random_forest": dict(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        scale_numeric=False,
        param_grid={
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 5, None],
            "clf__min_samples_leaf": [1, 3],
        },
    ),
    "xgboost": dict(
        estimator=XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        scale_numeric=False,
        param_grid={
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [2, 3, 4],
            "clf__learning_rate": [0.05, 0.1],
        },
    ),
}


def load_processed_data():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train, y_train = train.drop(columns=[TARGET_COLUMN]), train[TARGET_COLUMN]
    X_test, y_test = test.drop(columns=[TARGET_COLUMN]), test[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def train_and_compare(X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    """Cross-validated hyperparameter search per model, then a single
    held-out test evaluation per model (the test set is touched exactly
    once per model, after tuning is finished — tuning itself only ever
    sees cross-validation folds of the training set)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fitted_pipelines = {}

    for name, cfg in MODEL_CONFIGS.items():
        prep = build_preprocessor(scale_numeric=cfg["scale_numeric"])
        pipe = Pipeline([("prep", prep), ("clf", cfg["estimator"])])
        search = GridSearchCV(
            pipe, cfg["param_grid"], scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        fitted_pipelines[name] = best_pipe

        y_pred = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:, 1]

        results.append(
            {
                "model": name,
                "best_params": search.best_params_,
                "cv_roc_auc": round(search.best_score_, 4),
                "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
                "test_precision": round(precision_score(y_test, y_pred), 4),
                "test_recall": round(recall_score(y_test, y_pred), 4),
                "test_f1": round(f1_score(y_test, y_pred), 4),
                "test_roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            }
        )
        logger.info("%s: cv_roc_auc=%.4f test_roc_auc=%.4f test_recall=%.4f",
                     name, search.best_score_, results[-1]["test_roc_auc"], results[-1]["test_recall"])

    results_df = pd.DataFrame(results)
    return results_df, fitted_pipelines


def select_best_model(results_df: pd.DataFrame) -> str:
    """Selection rule: among models within 0.02 ROC-AUC of the best test
    ROC-AUC (i.e. statistically indistinguishable given a 60-row test set),
    pick the one with the highest recall. This operationalizes the
    "recall matters at least as much as raw accuracy" decision from the
    architecture doc, without ignoring ranking quality entirely."""
    best_auc = results_df["test_roc_auc"].max()
    contenders = results_df[results_df["test_roc_auc"] >= best_auc - 0.02]
    winner = contenders.sort_values("test_recall", ascending=False).iloc[0]
    return winner["model"]


def run(save: bool = True):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X_train, X_test, y_train, y_test = load_processed_data()
    results_df, fitted_pipelines = train_and_compare(X_train, y_train, X_test, y_test)

    print("\n=== Model comparison ===")
    print(
        results_df[
            ["model", "cv_roc_auc", "test_accuracy", "test_precision",
             "test_recall", "test_f1", "test_roc_auc"]
        ].to_string(index=False)
    )

    best_name = select_best_model(results_df)
    print(f"\nSelected model: {best_name}")

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted_pipelines[best_name], MODELS_DIR / "best_model.joblib")
        results_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
        logger.info("Saved best model (%s) and comparison table to %s", best_name, MODELS_DIR)

    return results_df, fitted_pipelines, best_name


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run()
