"""
tools/feature_labels.py
Single source of truth for "raw ML field name -> human-readable label",
shared by tools/risk_explanation.py (SHAP display) and
tools/patient_intake.py (follow-up acknowledgment text). Previously
duplicated as a private dict inside risk_explanation.py — pulled out once
a second tool needed the same mapping, rather than importing another
module's private name or re-typing the list.
"""

FEATURE_LABELS = {
    "age": "age",
    "sex": "sex",
    "cp": "chest pain type",
    "trestbps": "resting blood pressure",
    "chol": "cholesterol",
    "fbs": "fasting blood sugar",
    "restecg": "resting ECG result",
    "thalach": "maximum heart rate achieved",
    "exang": "exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "ST segment slope",
    "ca": "number of blocked vessels",
    "thal": "thalassemia result",
}

# Human-facing display order for patient-facing report sections —
# demographics first, then presenting symptoms, then vitals/labs, then
# test results. Deliberately NOT the same as
# tools.disease_prediction.FEATURE_COLUMNS, which is grouped by ML
# encoding type (nominal/binary/numeric, for the ColumnTransformer) —
# that order puts `thal` at index 3 and `age` at index 7, which reads
# fine to a pipeline and oddly to a human. Caught by generating a real
# sample report during P5 and noticing the field order looked arbitrary
# (it reflected the ML pipeline's internal grouping, not something a
# reader would expect from a clinical summary).
DISPLAY_ORDER = [
    "age", "sex",                                             # demographics
    "cp", "exang",                                             # presenting symptoms
    "trestbps", "chol", "fbs",                                 # vitals / labs
    "restecg", "thalach", "oldpeak", "slope", "ca", "thal",    # test results
]

# Human-readable meaning of each CATEGORICAL field's specific codes — see
# tools/validation.py CATEGORICAL_VALUES/BINARY_FIELDS for the authoritative
# valid-range definitions this must stay consistent with, and
# docs/data_audit_findings.md for how the thal 1/2/3 mapping was verified
# against the original UCI codebook (3/6/7, remapped in this dataset
# mirror). Numeric (non-categorical) fields are not listed here — their
# raw value IS their human-readable value (e.g. age=58, chol=260).
VALUE_LABELS = {
    "sex": {0: "female", 1: "male"},
    "cp": {0: "typical angina", 1: "atypical angina", 2: "non-anginal pain", 3: "asymptomatic (no chest pain)"},
    "fbs": {0: "no", 1: "yes"},
    "restecg": {0: "normal", 1: "ST-T wave abnormality", 2: "probable/definite left ventricular hypertrophy"},
    "exang": {0: "no", 1: "yes"},
    "slope": {0: "upsloping", 1: "flat", 2: "downsloping"},
    "thal": {1: "normal", 2: "fixed defect", 3: "reversible defect"},
}


def describe_value(base_feature: str, raw_value) -> str:
    """The single grounded description a narrative prompt should use for
    'what this patient's value actually is' — categorical fields map to
    their text meaning, numeric fields are shown as their number. This is
    what closes the gap where an LLM was asked to name a "specific
    finding" without ever being given one (see docs/agent_core_findings.md
    — the reversible-defect hallucination this fixes)."""
    if base_feature in VALUE_LABELS:
        try:
            return VALUE_LABELS[base_feature][int(raw_value)]
        except (KeyError, TypeError, ValueError):
            return str(raw_value)
    return str(raw_value)
