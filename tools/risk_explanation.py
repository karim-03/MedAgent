"""
tools/risk_explanation.py
Combines SHAP (exact, numeric — from P1) with an LLM narrative (fluent,
readable — from P3). The split is deliberate: SHAP values are the ground
truth, the LLM only puts them into plain language.

System prompt tightening #1 (P3 finding): the original benchmark prompt
said "state the numbers as given," which held up on numbers but slightly
over-generalized a categorical finding ("a thalassemia condition" instead
of "reversible defect"). Fixed by requiring the specific finding, not the
general category.

Fix #2, found on a real hardware run (docs/agent_core_findings.md) — more
serious than #1: telling the LLM to "name the SPECIFIC finding" while
never actually GIVING it the specific finding is a direct setup for
hallucination. The narrative prompt used to pass only a generic label
("thalassemia result") and a SHAP score — never the patient's actual
value. On a real run, the model confidently stated "reversible defect,"
which happened to be correct, but it was never told that; it invented a
plausible-sounding specific answer to comply with the "be specific"
instruction. get_shap_contributions() now looks up each top feature's
REAL value directly from the patient's own feature row (not inferred from
which one-hot dummy column SHAP happened to rank highest — that's a
separate, easy-to-misread signal; see the note in get_shap_contributions)
and build_narrative() passes that grounded value into the prompt. The LLM
now has something true to be specific about, instead of a reason to guess.
"""

from dataclasses import dataclass

from llm.client import LocalLLMClient
from ml.evaluation.shap_analysis import explain_single_prediction
from tools.disease_prediction import PredictionResult, _get_model
from tools.feature_labels import FEATURE_LABELS, describe_value

NARRATIVE_SYSTEM_PROMPT = """You are summarizing a machine learning model's heart disease risk prediction for a clinician. Rules, no exceptions:
1. State the probability exactly as given (e.g. 0.78 -> "78%"), never rounded to a vague band, never changed.
2. For each contributing factor, you will be given its exact value — state that value exactly (e.g. "reversible defect", "2 blocked vessels"). Never state a specific finding that was not given to you, and never generalize a given specific finding into a vaguer category.
3. Do not add clinical context, typical ranges, or interpretation that was not included in the input — describe only the exact values provided.
4. Write exactly 3 sentences."""


@dataclass
class RiskExplanation:
    shap_contributions: list  # list of {feature, specific_value, shap_value, ...} dicts
    narrative: str


def get_shap_contributions(prediction: PredictionResult, top_n: int = 3) -> list:
    """Pulls more raw SHAP rows than top_n and deduplicates by base
    feature before truncating. A one-hot-encoded categorical (e.g. thal
    has dummy columns thal_2, thal_3) can otherwise show up as the same
    "finding" twice in the top-N, which reads as a repeated/redundant
    explanation rather than the two distinct top features it should be.

    Important: which one-hot dummy column ranks highest in SHAP is NOT a
    reliable way to read off the patient's actual category (SHAP explains
    a tree ensemble's behavior across every dummy, including how much a
    dummy being OFF matters — the top-ranked dummy for a categorical
    feature is not necessarily the one matching this patient's true
    value). The base feature name from SHAP tells us WHICH feature
    matters; the ACTUAL value for narrative grounding is read directly
    from prediction.feature_row below, not guessed from the dummy name."""
    model = _get_model()
    # pull extra rows so dedup still has enough candidates to fill top_n
    contrib_df = explain_single_prediction(model, prediction.feature_row, "random_forest", top_n=top_n * 3)

    seen_bases = set()
    contributions = []
    for row in contrib_df.itertuples():
        base = row.feature.split("__", 1)[-1].rsplit("_", 1)[0]
        if base not in FEATURE_LABELS:
            base = row.feature.split("__", 1)[-1]
        if base in seen_bases:
            continue
        seen_bases.add(base)

        raw_value = prediction.feature_row[base].iloc[0] if base in prediction.feature_row.columns else None
        specific_value = describe_value(base, raw_value) if raw_value is not None else "unknown"

        contributions.append(
            {
                "feature": FEATURE_LABELS.get(base, base),
                "base_feature": base,
                "raw_feature": row.feature,
                "value": raw_value,
                "specific_value": specific_value,
                "shap_value": round(float(row.shap_value), 4),
            }
        )
        if len(contributions) == top_n:
            break
    return contributions


def build_narrative(client: LocalLLMClient, prediction: PredictionResult, shap_contributions: list) -> str:
    factors_text = "; ".join(
        f"{c['feature']} = {c['specific_value']} (SHAP={c['shap_value']:+.3f})" for c in shap_contributions
    )
    prompt = (
        f"Prediction: {'heart disease present' if prediction.predicted_class == 1 else 'heart disease absent'}, "
        f"probability {prediction.probability:.2f}. "
        f"Top contributing factors: {factors_text}."
    )
    result = client.generate(prompt=prompt, system=NARRATIVE_SYSTEM_PROMPT, json_format=False)
    return result.text.strip()


def explain(client: LocalLLMClient, prediction: PredictionResult, top_n: int = 3) -> RiskExplanation:
    contributions = get_shap_contributions(prediction, top_n=top_n)
    narrative = build_narrative(client, prediction, contributions)
    return RiskExplanation(shap_contributions=contributions, narrative=narrative)
