"""
tools/disease_prediction.py
Wraps the trained Random Forest pipeline from P1. This is the ONLY place
in the whole project allowed to produce a disease probability — the LLM
never does this (see docs/architecture.md Section 3.2 and the P3 findings
on why numeric fidelity from ML, not LLM narration, is what's trustworthy).

Takes already-validated, already-normalized fields (tools/validation.py's
job, not this tool's) and returns a probability plus the raw feature row,
so the Risk Explanation tool can run SHAP on the exact same row.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from config.loader import load_settings
from ml.training.preprocessing import NOMINAL_COLUMNS, BINARY_COLUMNS, NUMERIC_COLUMNS

# The 13 columns the pipeline's ColumnTransformer was fit on — derived
# from preprocessing.py's column groups (the ones actually used to build
# the trained pipeline) rather than a second hand-typed list that could
# silently drift out of sync with them. Order doesn't matter for a
# ColumnTransformer selecting by name; the SET must match exactly, or
# sklearn raises.
FEATURE_COLUMNS = NOMINAL_COLUMNS + BINARY_COLUMNS + NUMERIC_COLUMNS

_model_cache = None


def _get_model():
    global _model_cache
    if _model_cache is None:
        config = load_settings()
        model_path = Path(config["ml"]["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python scripts/run_p1_pipeline.py` first."
            )
        _model_cache = joblib.load(model_path)
    return _model_cache


@dataclass
class PredictionResult:
    probability: float
    predicted_class: int  # 1 = disease present, 0 = absent (see docs/data_audit_findings.md target-flip)
    feature_row: pd.DataFrame  # exactly what was fed to the model — Risk Explanation reuses this


def predict(normalized_fields: dict) -> PredictionResult:
    missing = [c for c in FEATURE_COLUMNS if c not in normalized_fields]
    if missing:
        raise ValueError(
            f"Cannot predict — missing required fields: {missing}. "
            "This tool must only be called after intake+validation confirm all fields are present."
        )

    row = pd.DataFrame([{col: normalized_fields[col] for col in FEATURE_COLUMNS}])
    model = _get_model()

    proba = model.predict_proba(row)[0, 1]
    predicted_class = int(proba >= 0.5)

    return PredictionResult(probability=float(proba), predicted_class=predicted_class, feature_row=row)
