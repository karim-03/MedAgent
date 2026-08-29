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
from tools.patient_intake import (
    select_next_missing_field, FIELD_PRIORITY_ORDER, _build_acknowledgment,
    FOLLOWUP_FIELD_TOPICS, generate_followup_question,
)
from tools.risk_explanation import get_shap_contributions, build_narrative
from tools.feature_labels import describe_value
from tools.knowledge_retrieval import build_queries

from conftest import COMPLETE_PATIENT


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


def test_followup_field_topics_covers_every_priority_field():
    assert set(FOLLOWUP_FIELD_TOPICS.keys()) == set(FIELD_PRIORITY_ORDER)


@pytest.mark.parametrize("field", list(FOLLOWUP_FIELD_TOPICS.keys()))
def test_followup_field_topics_contain_no_numeric_codes(field):
    """Regression test for a real leak found on a hardware run: the
    follow-up question for `thal` included '(1)', '(2)', '(3)' because the
    prompt reused the coded EXTRACTION field description. The actual leak
    pattern is a bare category code — "(1)" or "1=normal" — not any digit
    anywhere; legitimate clinical numbers like "120 mg/dl" in the fbs
    topic are fine and expected, not a regression."""
    import re
    topic = FOLLOWUP_FIELD_TOPICS[field]
    assert not re.search(r"\(\d+\)", topic), f"{field!r} topic leaks a parenthetical code: {topic!r}"
    assert not re.search(r"\b\d+\s*=", topic), f"{field!r} topic leaks a 'N=' style code: {topic!r}"


def test_generate_followup_question_prompt_has_no_field_description_codes():
    """End-to-end check that the prompt actually sent to the LLM (not
    just the topic dict) is code-free — catches a regression even if
    someone reintroduces FIELD_DESCRIPTIONS parsing here later."""
    class RecordingClient:
        def generate(self, prompt, system=None, json_format=False, options=None):
            self.last_prompt = prompt
            from llm.client import GenerationResult
            return GenerationResult(
                text="fake question?", model_used="fake", used_fallback=False,
                prompt_tokens=1, completion_tokens=1, total_duration_s=0.01, tokens_per_second=1.0,
            )

    client = RecordingClient()
    generate_followup_question(client, "thal")
    assert not any(char.isdigit() for char in client.last_prompt)


# ---------- feature_labels.describe_value ----------

@pytest.mark.parametrize("field,value,expected", [
    ("thal", 1, "normal"),
    ("thal", 2, "fixed defect"),
    ("thal", 3, "reversible defect"),
    ("cp", 3, "asymptomatic (no chest pain)"),
    ("exang", 1, "yes"),
    ("exang", 0, "no"),
    ("sex", 1, "male"),
])
def test_describe_value_maps_categorical_codes_to_meaning(field, value, expected):
    assert describe_value(field, value) == expected


def test_describe_value_passes_through_numeric_fields_as_is():
    assert describe_value("age", 58) == "58"
    assert describe_value("ca", 2) == "2"


def test_describe_value_handles_unexpected_value_gracefully():
    # out-of-codebook value shouldn't crash — validation.py is the layer
    # responsible for rejecting it; this function just needs to not blow up
    assert describe_value("thal", 99) == "99"


# ---------- risk_explanation.get_shap_contributions / build_narrative ----------
# Regression coverage for a real bug found on a hardware run: the
# narrative LLM was told to "name the specific finding" without ever being
# given one, and fabricated a plausible-sounding answer. These tests check
# the fix — that the ACTUAL patient value is looked up and grounded,
# independent of which one-hot dummy column SHAP happens to rank highest.
# Uses the shared COMPLETE_PATIENT fixture — thal=3 there is already
# "reversible defect", which is exactly the case these tests need.


def test_shap_contributions_ground_thal_to_patients_actual_value_not_top_dummy():
    """Direct regression test: for this exact patient (thal=3, reversible
    defect), the top-ranked SHAP one-hot dummy is 'cat__thal_2' — NOT
    'cat__thal_3' — because SHAP attributes importance per dummy column,
    not per true category. Grounding must come from the real feature row,
    not from parsing which dummy ranked highest."""
    result = validate_patient_fields(COMPLETE_PATIENT)
    prediction = predict(result.normalized_fields)
    contributions = get_shap_contributions(prediction)

    thal_contrib = next(c for c in contributions if c["base_feature"] == "thal")
    assert thal_contrib["value"] == 3
    assert thal_contrib["specific_value"] == "reversible defect"
    # the known one-hot quirk this test guards against — if this ever
    # stops being true, the test is still correct either way since it
    # asserts specific_value directly, not the dummy column name
    assert thal_contrib["raw_feature"] != "cat__thal_3"


def test_narrative_prompt_includes_grounded_value_not_just_label():
    """build_narrative must give the LLM something true to be specific
    about — checks the actual prompt sent, not just the final text,
    since a real LLM might produce plausible output either way and mask
    whether grounding actually happened."""
    class RecordingClient:
        def __init__(self):
            self.last_prompt = None

        def generate(self, prompt, system=None, json_format=False, options=None):
            self.last_prompt = prompt
            from llm.client import GenerationResult
            return GenerationResult(
                text="fake narrative", model_used="fake", used_fallback=False,
                prompt_tokens=1, completion_tokens=1, total_duration_s=0.01, tokens_per_second=1.0,
            )

    result = validate_patient_fields(COMPLETE_PATIENT)
    prediction = predict(result.normalized_fields)
    contributions = get_shap_contributions(prediction)

    client = RecordingClient()
    build_narrative(client, prediction, contributions)

    assert "thalassemia result = reversible defect" in client.last_prompt


# ---------- knowledge_retrieval.build_queries ----------
# Regression coverage for a real, recurring issue: every P4 hardware run
# showed an irrelevant "blood pressure categories" passage riding along
# as filler evidence, because one query asked for k=2 results instead of
# querying each top feature's own topic separately.

SAMPLE_SHAP_CONTRIBUTIONS = [
    {"feature": "thalassemia result", "base_feature": "thal", "raw_feature": "cat__thal_2", "specific_value": "reversible defect", "shap_value": 0.0842},
    {"feature": "number of blocked vessels", "base_feature": "ca", "raw_feature": "num__ca", "specific_value": "2", "shap_value": 0.0688},
    {"feature": "exercise-induced angina", "base_feature": "exang", "raw_feature": "bin__exang", "specific_value": "yes", "shap_value": 0.0521},
]


def test_build_queries_returns_one_query_per_contribution():
    queries = build_queries(SAMPLE_SHAP_CONTRIBUTIONS)
    assert len(queries) == 3
    assert queries[0] != queries[1] != queries[2]  # distinct topics, not the same query repeated


def test_build_queries_respects_max_queries():
    queries = build_queries(SAMPLE_SHAP_CONTRIBUTIONS, max_queries=1)
    assert len(queries) == 1


def test_build_queries_falls_back_on_empty_contributions():
    queries = build_queries([])
    assert queries == ["heart disease risk factors"]


def test_build_queries_uses_base_feature_directly_not_raw_feature_parsing():
    # confirms the fix uses the already-computed base_feature field rather
    # than re-parsing raw_feature — the thal query should reflect 'thal',
    # not whatever base cat__thal_2's suffix-stripping would produce
    queries = build_queries([SAMPLE_SHAP_CONTRIBUTIONS[0]])
    assert "thalassemia" in queries[0].lower()
