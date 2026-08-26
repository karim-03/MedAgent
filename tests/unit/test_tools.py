"""
Unit tests for tools/validation.py and tools/disease_prediction.py — both
are pure Python / real trained model, no LLM or network needed, so these
run for real every time, no skip guards required.

Run with: pytest tests/unit/test_tools.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.validation import validate_patient_fields, missing_required_fields
from tools.disease_prediction import predict, FEATURE_COLUMNS
from tools.patient_intake import select_next_missing_field, FIELD_PRIORITY_ORDER, _build_acknowledgment

COMPLETE_PATIENT = {
    "age": 58, "sex": "man", "cp": 3, "trestbps": 145, "chol": 260,
    "fbs": "no", "restecg": 0, "thalach": 132, "exang": "yes",
    "oldpeak": 2.1, "slope": 1, "ca": 2, "thal": 3,
}


# ---------- validation.py ----------

@pytest.mark.parametrize("value,expected", [
    ("man", 1), ("Male", 1), ("M", 1), (1, 1),
    ("woman", 0), ("Female", 0), ("F", 0), (0, 0),
])
def test_sex_normalizes_correctly(value, expected):
    result = validate_patient_fields({"sex": value})
    assert result.normalized_fields["sex"] == expected
    assert result.is_valid


def test_unrecognized_sex_value_is_an_error_not_a_silent_guess():
    result = validate_patient_fields({"sex": "unspecified"})
    assert not result.is_valid
    assert "sex" not in result.normalized_fields


@pytest.mark.parametrize("value,expected", [("yes", 1), ("no", 0), ("Y", 1), ("N", 0)])
def test_yes_no_fields_normalize_correctly(value, expected):
    result = validate_patient_fields({"exang": value})
    assert result.normalized_fields["exang"] == expected


def test_out_of_range_numeric_value_is_rejected():
    result = validate_patient_fields({**COMPLETE_PATIENT, "trestbps": 9999})
    assert not result.is_valid
    assert any("trestbps" in e for e in result.errors)


def test_out_of_codebook_categorical_value_is_rejected():
    result = validate_patient_fields({**COMPLETE_PATIENT, "cp": 7})
    assert not result.is_valid
    assert any("cp" in e for e in result.errors)


def test_complete_valid_patient_has_no_errors():
    result = validate_patient_fields(COMPLETE_PATIENT)
    assert result.is_valid
    assert result.errors == []


def test_missing_required_fields_detects_gaps():
    result = validate_patient_fields({"age": 58, "sex": "male"})
    missing = missing_required_fields(result.normalized_fields, ["age", "sex", "cp", "chol"])
    assert missing == ["cp", "chol"]


def test_missing_required_fields_empty_when_all_present():
    result = validate_patient_fields(COMPLETE_PATIENT)
    missing = missing_required_fields(result.normalized_fields, list(COMPLETE_PATIENT.keys()))
    assert missing == []


# ---------- disease_prediction.py ----------

def test_predict_returns_probability_in_valid_range():
    result = validate_patient_fields(COMPLETE_PATIENT)
    prediction = predict(result.normalized_fields)
    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.predicted_class in (0, 1)


def test_predict_feature_row_matches_expected_columns():
    result = validate_patient_fields(COMPLETE_PATIENT)
    prediction = predict(result.normalized_fields)
    assert list(prediction.feature_row.columns) == FEATURE_COLUMNS
    assert len(prediction.feature_row) == 1


def test_predict_raises_clear_error_on_incomplete_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        predict({"age": 58, "sex": 1})


def test_predict_is_deterministic_for_same_input():
    result = validate_patient_fields(COMPLETE_PATIENT)
    p1 = predict(result.normalized_fields)
    p2 = predict(result.normalized_fields)
    assert p1.probability == p2.probability


# ---------- patient_intake.select_next_missing_field ----------

def test_select_next_missing_field_picks_highest_priority():
    missing = ["fbs", "thal", "age", "ca"]  # deliberately out of priority order
    assert select_next_missing_field(missing) == "thal"


def test_select_next_missing_field_is_deterministic_across_calls():
    missing = ["chol", "exang", "restecg"]
    results = {select_next_missing_field(missing) for _ in range(5)}
    assert results == {"exang"}  # exang outranks chol/restecg — same answer every time, no randomness


def test_select_next_missing_field_returns_none_when_nothing_missing():
    assert select_next_missing_field([]) is None


def test_select_next_missing_field_handles_unranked_field_gracefully():
    # every real field IS in FIELD_PRIORITY_ORDER, but the function
    # shouldn't crash if it's ever called with something outside the schema
    result = select_next_missing_field(["some_unknown_field", "thal"])
    assert result == "thal"  # ranked field still wins over an unranked one


def test_field_priority_order_covers_every_required_field():
    # guards against silently forgetting to add a field to the priority
    # list if the ML schema ever grows
    required = {
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    }
    assert set(FIELD_PRIORITY_ORDER) == required


# ---------- patient_intake._build_acknowledgment ----------

def test_build_acknowledgment_single_field():
    text = _build_acknowledgment({"chol": 260})
    assert text == "Thanks — got your cholesterol."


def test_build_acknowledgment_multiple_fields_joined_naturally():
    text = _build_acknowledgment({"chol": 260, "fbs": 0})
    assert text == "Thanks — got your cholesterol and fasting blood sugar."


def test_build_acknowledgment_uses_human_labels_not_raw_field_codes():
    text = _build_acknowledgment({"thal": 3})
    assert "thal" not in text.lower().split()  # raw code shouldn't leak in
    assert "thalassemia" in text.lower()
