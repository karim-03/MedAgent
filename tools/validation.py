"""
tools/validation.py
Merges the original spec's Input Validator + Symptom Validator into one
tool, per the MVP scope cut agreed in docs/architecture.md Section 8.3.

Two jobs, always in this order:
1. Normalize free-text/LLM-extracted values into the exact codes the ML
   model was trained on (e.g. "man" -> 1, "yes" -> 1) — this directly
   closes the finding from docs/llm_verification_findings.md, where the
   intake LLM correctly extracted `"sex": "man"` but that's not a value
   the trained pipeline's ColumnTransformer was ever fit on.
2. Range/type-check the normalized values against the codebook (see
   docs/data_audit_findings.md) and reject anything outside it, so a bad
   value never silently reaches the ML model as a "valid" input.

This tool is pure Python — no LLM, no network — so it's fully unit
testable and deterministic, which matters for a clinical-adjacent input
boundary.
"""

from dataclasses import dataclass, field

# Codebook ranges — see docs/data_audit_findings.md for where these numbers
# come from (the UCI codebook + the audit's cleaning decisions).
NUMERIC_RANGES = {
    "age": (1, 120),
    "trestbps": (60, 250),   # resting blood pressure, mm Hg
    "chol": (100, 700),      # serum cholesterol, mg/dl
    "thalach": (60, 250),    # max heart rate achieved
    "oldpeak": (0.0, 10.0),  # ST depression
    "ca": (0, 3),            # number of major vessels (codebook valid range)
}
CATEGORICAL_VALUES = {
    "cp": {0, 1, 2, 3},        # chest pain type
    "restecg": {0, 1, 2},      # resting ECG results
    "slope": {0, 1, 2},        # slope of peak exercise ST segment
    "thal": {1, 2, 3},         # thalassemia (codebook valid range — see audit findings re: 0 being a placeholder)
}
BINARY_FIELDS = {"sex", "fbs", "exang"}

# Free-text -> code normalization. Deliberately explicit and small rather
# than "clever" fuzzy matching — a wrong silent guess here is worse than
# asking the user to rephrase.
_SEX_MAP = {
    "man": 1, "male": 1, "m": 1, "1": 1, 1: 1,
    "woman": 0, "female": 0, "f": 0, "0": 0, 0: 0,
}
_YES_NO_MAP = {
    "yes": 1, "true": 1, "y": 1, "1": 1, 1: 1,
    "no": 0, "false": 0, "n": 0, "0": 0, 0: 0,
}


@dataclass
class ValidationResult:
    is_valid: bool
    normalized_fields: dict
    errors: list = field(default_factory=list)


def _normalize_value(key: str, value):
    if value is None:
        return None, None  # missing, not invalid — intake tool's job to ask for it

    if key == "sex":
        norm = _SEX_MAP.get(str(value).strip().lower())
        if norm is None:
            return None, f"Could not interpret sex value: {value!r}"
        return norm, None

    if key in ("fbs", "exang"):
        norm = _YES_NO_MAP.get(str(value).strip().lower())
        if norm is None:
            return None, f"Could not interpret yes/no value for {key}: {value!r}"
        return norm, None

    if key in NUMERIC_RANGES:
        try:
            norm = float(value)
            if key != "oldpeak":
                norm = int(norm)
        except (TypeError, ValueError):
            return None, f"{key} must be a number, got {value!r}"
        return norm, None

    if key in CATEGORICAL_VALUES:
        try:
            norm = int(value)
        except (TypeError, ValueError):
            return None, f"{key} must be an integer code, got {value!r}"
        return norm, None

    return value, f"Unknown field: {key}"


def validate_patient_fields(raw_fields: dict) -> ValidationResult:
    normalized = {}
    errors = []

    for key, value in raw_fields.items():
        norm_value, err = _normalize_value(key, value)
        if err:
            errors.append(err)
            continue
        if norm_value is None:
            continue  # missing field, not an error at this layer
        normalized[key] = norm_value

    for key, value in normalized.items():
        if key in NUMERIC_RANGES:
            lo, hi = NUMERIC_RANGES[key]
            if not (lo <= value <= hi):
                errors.append(f"{key}={value} is outside the plausible range [{lo}, {hi}]")
        elif key in CATEGORICAL_VALUES:
            if value not in CATEGORICAL_VALUES[key]:
                errors.append(f"{key}={value} is not one of the valid codes {sorted(CATEGORICAL_VALUES[key])}")
        elif key in BINARY_FIELDS:
            if value not in (0, 1):
                errors.append(f"{key}={value} must normalize to 0 or 1")

    return ValidationResult(is_valid=len(errors) == 0, normalized_fields=normalized, errors=errors)


def missing_required_fields(normalized_fields: dict, required_fields: list) -> list:
    return [f for f in required_fields if f not in normalized_fields or normalized_fields[f] is None]
