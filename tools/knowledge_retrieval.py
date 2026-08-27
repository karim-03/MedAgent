"""
tools/knowledge_retrieval.py
Turns the top SHAP-contributing features into retrieval queries against
the P2 knowledge base, so the evidence shown to the clinician is actually
about the specific factors that drove THIS patient's prediction — not a
generic "heart disease" query.

Design fix, found by actually reading the evidence on every P4 hardware
run: the first version built ONE query from only the single top feature,
then asked for k=2 results from it. For a small, single-topic-sparse
corpus, that means the second result is whatever's second-closest in the
ENTIRE corpus to that one query — not necessarily relevant at all. Every
real run showed the same irrelevant "Blood pressure categories" passage
riding along as filler evidence for a thalassemia finding. The fix isn't a
higher relevance threshold (that would just return fewer results, not
better ones) — it's asking a separate, topically-focused question per top
feature, so a second passage exists because a second genuinely relevant
topic exists (e.g. blocked vessels), not because k=2 demanded a second
result regardless.
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


def build_queries(shap_contributions: list, max_queries: int = 3) -> list:
    """One topically-focused query per top contributing feature (not one
    query reused for multiple results) — see module docstring for why."""
    queries = []
    for contrib in shap_contributions[:max_queries]:
        base = contrib.get("base_feature")
        if not base or base not in _FEATURE_QUERY_TOPICS:
            raw = contrib.get("raw_feature", "")
            base = raw.split("__", 1)[-1].rsplit("_", 1)[0]
        queries.append(_FEATURE_QUERY_TOPICS.get(base, "heart disease risk factors"))
    return queries or ["heart disease risk factors"]


@dataclass
class RetrievalOutcome:
    queries: list  # one per top contributing feature
    passages: list  # list[RetrievedPassage], deduplicated — empty if nothing cleared the relevance threshold


def retrieve_evidence(shap_contributions: list, k_per_query: int = 1) -> RetrievalOutcome:
    queries = build_queries(shap_contributions)
    seen_texts = set()
    passages = []
    for q in queries:
        for passage in retrieve(q, k=k_per_query):
            if passage.text not in seen_texts:
                seen_texts.add(passage.text)
                passages.append(passage)
    return RetrievalOutcome(queries=queries, passages=passages)
