"""
Unit tests for tools/report_generator.py — fully real, no mocking needed:
the trained model, real SHAP contributions, real config-driven risk tiers.
The only input that isn't produced by a real tool call is the narrative
text, which report_generator.py deliberately just consumes as a string
(it doesn't call the LLM itself) — a canned string is exactly as real a
test input as an LLM-generated one would be, from this tool's perspective.

Run with: pytest tests/unit/test_report_generator.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.validation import validate_patient_fields
from tools.disease_prediction import predict
from tools.risk_explanation import get_shap_contributions
from tools.report_generator import (
    generate_report, render_chat_summary, render_full_report_markdown,
    save_report_markdown, _get_risk_tier,
)

from conftest import COMPLETE_PATIENT

SAMPLE_NARRATIVE = (
    "The model predicts a 88% probability of heart disease. Top factors: "
    "a reversible defect, two blocked vessels, and exercise-induced angina."
)


@pytest.fixture(scope="module")
def real_report():
    result = validate_patient_fields(COMPLETE_PATIENT)
    prediction = predict(result.normalized_fields)
    contributions = get_shap_contributions(prediction)
    return generate_report(
        normalized_fields=result.normalized_fields,
        probability=prediction.probability,
        predicted_class=prediction.predicted_class,
        narrative=SAMPLE_NARRATIVE,
        shap_contributions=contributions,
        retrieved_passages=[],
    )


# ---------- _get_risk_tier ----------

@pytest.mark.parametrize("probability,expected_label", [
    (0.1, "Lower estimated risk"),
    (0.4, "Lower estimated risk"),   # boundary — inclusive on the lower tier
    (0.41, "Moderate estimated risk"),
    (0.7, "Moderate estimated risk"),  # boundary — inclusive on the moderate tier
    (0.71, "Higher estimated risk"),
    (1.0, "Higher estimated risk"),
])
def test_risk_tier_boundaries(probability, expected_label):
    assert _get_risk_tier(probability).label == expected_label


def test_risk_tier_always_returns_a_recommendation():
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        tier = _get_risk_tier(p)
        assert tier.recommendation.strip()


# ---------- generate_report ----------

def test_generate_report_uses_real_prediction_and_shap(real_report):
    assert 0.0 <= real_report.probability <= 1.0
    assert real_report.predicted_class in (0, 1)
    assert len(real_report.shap_contributions) == 3
    assert real_report.risk_tier.label  # populated, not empty


def test_generate_report_disclaimer_matches_config(real_report):
    from config.loader import load_settings
    expected = load_settings()["reporting"]["disclaimer"].strip()
    assert real_report.disclaimer == expected


def test_generate_report_handles_empty_evidence_list(real_report):
    assert real_report.retrieved_passages == []


# ---------- render_chat_summary ----------

def test_chat_summary_includes_narrative_and_disclaimer(real_report):
    summary = render_chat_summary(real_report)
    assert SAMPLE_NARRATIVE in summary
    assert real_report.disclaimer in summary


def test_chat_summary_notes_when_no_evidence_found(real_report):
    summary = render_chat_summary(real_report)
    assert "No directly relevant passage" in summary


# ---------- render_full_report_markdown ----------

REQUIRED_SECTIONS = [
    "## Patient Information",
    "## Prediction",
    "## Explanation",
    "## Top Contributing Factors",
    "## Supporting Evidence",
    "## Recommended Follow-Up",
]


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_full_report_contains_every_required_section(real_report, section):
    """Direct coverage of the project spec's REPORTS requirement: every
    one of these sections must be present, not just some of them."""
    report_text = render_full_report_markdown(real_report)
    assert section in report_text


def test_full_report_disclaimer_appears_at_start_and_end(real_report):
    report_text = render_full_report_markdown(real_report)
    assert report_text.count(real_report.disclaimer) >= 1
    # appears both as the leading block-quote and the closing line
    assert report_text.index(real_report.disclaimer) < len(report_text) / 2


def test_full_report_patient_summary_uses_human_labels_not_raw_codes(real_report):
    report_text = render_full_report_markdown(real_report)
    assert "thalassemia result" in report_text
    assert "reversible defect" in report_text
    # the raw ML field name should not leak into patient-facing text
    assert "**thal**" not in report_text


def test_patient_summary_is_in_canonical_order_not_conversation_order():
    """Regression test: normalized_fields accumulates via dict-merging
    across conversation turns, so its insertion order reflects whatever
    order the patient happened to mention things in — a real conversation
    could easily produce thal before age. The rendered summary must not
    follow that order; it must always read in a consistent clinical
    field order regardless of how the dict was built."""
    out_of_order_fields = {
        "thal": 3, "ca": 2, "exang": 1, "age": 58, "sex": 1,
        "cp": 3, "trestbps": 145, "chol": 260, "fbs": 0,
        "restecg": 0, "thalach": 132, "oldpeak": 2.1, "slope": 1,
    }
    report = generate_report(
        normalized_fields=out_of_order_fields, probability=0.5, predicted_class=1,
        narrative="x", shap_contributions=[], retrieved_passages=[],
    )
    report_text = render_full_report_markdown(report)
    age_pos = report_text.index("**age**")
    thal_pos = report_text.index("**thalassemia result**")
    assert age_pos < thal_pos, "patient summary should list age before thal regardless of dict order"


def test_full_report_shows_shap_direction_and_value(real_report):
    report_text = render_full_report_markdown(real_report)
    assert "SHAP=" in report_text
    assert "increases predicted risk" in report_text or "decreases predicted risk" in report_text


def test_full_report_includes_test_set_size_caveat(real_report):
    """Regression guard for a real project finding (P1): don't overclaim
    precision from a 60-patient test set — the report must say so."""
    report_text = render_full_report_markdown(real_report)
    assert "60-patient test set" in report_text


# ---------- save_report_markdown ----------

def test_save_report_markdown_writes_real_file(real_report, tmp_path):
    out_path = tmp_path / "test_report.md"
    result_path = save_report_markdown(real_report, path=out_path)
    assert result_path == out_path
    assert out_path.exists()
    assert "## Prediction" in out_path.read_text(encoding="utf-8")


def test_save_report_markdown_default_path_is_timestamped(real_report, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result_path = save_report_markdown(real_report)
    assert result_path.parent == Path("outputs/reports")
    assert result_path.exists()
    assert result_path.suffix == ".md"
