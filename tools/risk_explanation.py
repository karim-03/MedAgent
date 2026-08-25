"""
tools/risk_explanation.py
Combines SHAP (exact, numeric — from P1) with an LLM narrative (fluent,
readable — from P3). The split is deliberate: SHAP values are the ground
truth, the LLM only puts them into plain language.

System prompt below is the P3-finding-driven tightening: the original
benchmark prompt said "state the numbers as given," which held up on
numbers but slightly over-generalized a categorical finding ("a
thalassemia condition" instead of "reversible defect"). This version is
explicit about preserving the specific finding type, not just the general
category, and about not adding outside clinical context.
"""

from dataclasses import dataclass

from llm.client import LocalLLMClient
from ml.evaluation.shap_analysis import explain_single_prediction
from tools.disease_prediction import PredictionResult, _get_model

NARRATIVE_SYSTEM_PROMPT = """You are summarizing a machine learning model's heart disease risk prediction for a clinician. Rules, no exceptions:
1. State the probability exactly as given (e.g. 0.78 -> "78%"), never rounded to a vague band, never changed.
2. For each contributing factor, name the SPECIFIC finding given (e.g. "reversible defect", not the general category "a thalassemia condition"; "2 blocked vessels", not "several vessels" or "up to 3").
3. Do not add clinical context, typical ranges, or interpretation that was not included in the input — describe only the exact values provided.
4. Write exactly 3 sentences."""


@dataclass
class RiskExplanation:
    shap_contributions: list  # list of {feature, shap_value} dicts, human-labeled
    narrative: str


# Maps a raw ML feature name to (a) a human label for SHAP display and
# (b) the specific-value phrase used in the narrative prompt.
_FEATURE_LABELS = {
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


def _humanize_shap_feature_name(transformed_name: str) -> str:
    """SHAP operates on post-one-hot-encoding column names like
    'cat__thal_2' or 'num__ca' — map back to the raw feature for display."""
    base = transformed_name.split("__", 1)[-1]
    base = base.rsplit("_", 1)[0] if base.rsplit("_", 1)[0] in _FEATURE_LABELS else base
    return _FEATURE_LABELS.get(base, base)


def get_shap_contributions(prediction: PredictionResult, top_n: int = 3) -> list:
    """Pulls more raw SHAP rows than top_n and deduplicates by base
    feature before truncating. A one-hot-encoded categorical (e.g. thal
    has dummy columns thal_2, thal_3) can otherwise show up as the same
    "finding" twice in the top-N, which reads as a repeated/redundant
    explanation rather than the two distinct top features it should be."""
    model = _get_model()
    # pull extra rows so dedup still has enough candidates to fill top_n
    contrib_df = explain_single_prediction(model, prediction.feature_row, "random_forest", top_n=top_n * 3)

    seen_bases = set()
    contributions = []
    for row in contrib_df.itertuples():
        base = row.feature.split("__", 1)[-1].rsplit("_", 1)[0]
        if base not in _FEATURE_LABELS:
            base = row.feature.split("__", 1)[-1]
        if base in seen_bases:
            continue
        seen_bases.add(base)
        contributions.append(
            {
                "feature": _humanize_shap_feature_name(row.feature),
                "raw_feature": row.feature,
                "shap_value": round(float(row.shap_value), 4),
            }
        )
        if len(contributions) == top_n:
            break
    return contributions


def build_narrative(client: LocalLLMClient, prediction: PredictionResult, shap_contributions: list) -> str:
    factors_text = "; ".join(f"{c['feature']} (SHAP={c['shap_value']:+.3f})" for c in shap_contributions)
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
