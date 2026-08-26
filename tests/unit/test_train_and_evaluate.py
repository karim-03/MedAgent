"""
Unit tests for ml/training/train.py and ml/evaluation/*.py

These deliberately avoid re-running the full GridSearchCV (slow, and not
the point of a unit test) — they use small, fixed-parameter models to check
that the plumbing (preprocessing -> pipeline -> metrics -> SHAP) is correct.
Run with: pytest tests/unit/test_train_and_evaluate.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.training.train import build_preprocessor, load_processed_data, select_best_model
from ml.evaluation.metrics import plot_confusion_matrix, plot_feature_importance
from ml.evaluation.shap_analysis import explain_single_prediction


@pytest.fixture(scope="module")
def data():
    return load_processed_data()


@pytest.fixture(scope="module")
def small_rf_pipeline(data):
    X_train, X_test, y_train, y_test = data
    prep = build_preprocessor(scale_numeric=False)
    pipe = Pipeline([
        ("prep", prep),
        ("clf", RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def test_preprocessor_output_has_no_nans(data):
    X_train, X_test, y_train, y_test = data
    prep = build_preprocessor(scale_numeric=False)
    transformed = prep.fit_transform(X_train)
    assert not pd.DataFrame(transformed).isnull().values.any()


def test_pipeline_predicts_valid_probabilities(small_rf_pipeline, data):
    X_train, X_test, y_train, y_test = data
    proba = small_rf_pipeline.predict_proba(X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert len(proba) == len(X_test)


def test_select_best_model_prefers_higher_recall_within_auc_tolerance():
    results_df = pd.DataFrame([
        {"model": "a", "test_roc_auc": 0.96, "test_recall": 0.80},
        {"model": "b", "test_roc_auc": 0.95, "test_recall": 0.90},  # within 0.02 of best AUC
        {"model": "c", "test_roc_auc": 0.80, "test_recall": 0.99},  # too far below best AUC
    ])
    assert select_best_model(results_df) == "b"


def test_confusion_matrix_shape(small_rf_pipeline, data, tmp_path, monkeypatch):
    X_train, X_test, y_train, y_test = data
    y_pred = small_rf_pipeline.predict(X_test)
    monkeypatch.chdir(tmp_path)  # don't pollute the real outputs/ dir during tests
    cm = plot_confusion_matrix(y_test, y_pred, "test_model")
    assert cm.shape == (2, 2)
    assert (tmp_path / "outputs" / "figures" / "confusion_matrix_test_model.png").exists()


def test_feature_importance_returns_expected_columns(small_rf_pipeline, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    imp_df = plot_feature_importance(small_rf_pipeline, "test_model", top_n=5)
    assert list(imp_df.columns) == ["feature", "importance"]
    assert len(imp_df) == 5
    assert (imp_df["importance"] >= 0).all()


def test_explain_single_prediction_returns_signed_contributions(small_rf_pipeline, data):
    X_train, X_test, y_train, y_test = data
    one_patient = X_test.iloc[[0]]
    contrib = explain_single_prediction(small_rf_pipeline, one_patient, "test_model", top_n=3)
    assert len(contrib) == 3
    assert set(contrib.columns) == {"feature", "shap_value"}
