"""
agent/state.py
Shared state object threaded through every node in the LangGraph graph.
This IS the "conversation memory" for P4 — a running record of what's been
extracted, validated, and computed so far, not a separate memory subsystem.
"""

from typing import Optional, TypedDict

from tools.disease_prediction import PredictionResult


class AgentState(TypedDict, total=False):
    messages: list             # [{"role": "user"|"assistant", "content": str}, ...]
    extracted_fields: dict     # raw, pre-validation fields gathered so far
    newly_learned_fields: dict # subset of extracted_fields first learned THIS turn — lets follow-up questions acknowledge what was just said
    normalized_fields: dict    # post-validation, ML-ready fields
    validation_errors: list
    missing_fields: list
    followup_count: int        # guarded by agent.max_followup_questions in config

    prediction_probability: Optional[float]
    prediction_class: Optional[int]
    prediction_result: Optional[PredictionResult]  # full object (incl. feature row) — SHAP needs it, not just the probability

    shap_contributions: Optional[list]
    narrative: Optional[str]

    retrieval_query: Optional[str]
    retrieved_passages: Optional[list]

    turn_response: Optional[str]  # what the agent says back this turn
    done: bool                     # True once prediction+explanation+evidence are all in hand
