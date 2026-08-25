"""
tools/knowledge_retrieval.py
Turns the top SHAP-contributing features into a retrieval query against
the P2 knowledge base, so the evidence shown to the clinician is actually
about the specific factors that drove THIS patient's prediction — not a
generic "heart disease" query.
"""

from dataclasses import dataclass

from rag.retrieve import RetrievedPassage, retrieve

# Maps a raw ML feature to the natural-language topic phrase that best
# matches how the P2 knowledge base documents actually discuss it (see
# docs/knowledge_base_findings.md for which document covers which feature).
_FEATURE_QUERY_TOPICS = {
    "age": "age as a heart disease risk factor",
    "sex": "sex differences in heart disease risk",
    "cp": "chest pain and angina",
    "trestbps": "high blood pressure diagnosis",
    "chol": "cholesterol levels and heart disease risk",
    "fbs": "diabetes and heart disease risk",
    "restecg": "electrocardiogram ECG results",
    "thalach": "maximum heart rate during exercise stress test",
    "exang": "exercise-induced angina",
    "oldpeak": "ST depression exercise stress test",
    "slope": "exercise stress test ECG findings",
    "ca": "coronary angiography blocked vessels",
    "thal": "thalassemia heart test result",
}


def build_query(shap_contributions: list) -> str:
    """Uses only the single top contributing feature — a focused query
    retrieves a more relevant passage than a broad multi-topic one."""
    if not shap_contributions:
        return "heart disease risk factors"
    top_feature = shap_contributions[0]["raw_feature"]
    base = top_feature.split("__", 1)[-1].rsplit("_", 1)[0]
    if base not in _FEATURE_QUERY_TOPICS:
        base = top_feature.split("__", 1)[-1]
    return _FEATURE_QUERY_TOPICS.get(base, "heart disease risk factors")


@dataclass
class RetrievalOutcome:
    query: str
    passages: list  # list[RetrievedPassage] — empty if nothing cleared the relevance threshold


def retrieve_evidence(shap_contributions: list, k: int = 2) -> RetrievalOutcome:
    query = build_query(shap_contributions)
    passages = retrieve(query, k=k)
    return RetrievalOutcome(query=query, passages=passages)
