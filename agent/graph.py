"""
agent/graph.py
The LangGraph state machine implementing the workflow from
docs/architecture.md Section 4: parse -> (loop on missing/invalid info) ->
predict -> explain -> retrieve evidence -> synthesize.

Built as a factory (`build_graph(client)`) taking a dependency-injected
LLM client, rather than each node reaching for a global — matches the
project's stated preference for DI where it's useful, and makes the graph
testable with a fake client (see tests/unit/test_agent_graph.py).
"""

import logging
from pathlib import Path

import yaml
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from llm.client import LocalLLMClient
from tools import patient_intake, validation, disease_prediction, risk_explanation, knowledge_retrieval

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/settings.yaml")
DISCLAIMER = (
    "This is an educational capstone project, not a certified medical device. "
    "It is not validated for clinical use and must never be used to make real "
    "patient-care decisions. Please consult a qualified healthcare professional."
)


def _load_agent_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["agent"]


def build_graph(client: LocalLLMClient):
    agent_config = _load_agent_config()
    required_fields = agent_config["required_fields"]
    max_followups = agent_config["max_followup_questions"]

    def intake_node(state: AgentState) -> dict:
        last_user_message = state["messages"][-1]["content"]
        extraction = patient_intake.extract_fields_from_message(client, last_user_message)

        previously_known = state.get("extracted_fields", {})
        merged = {**previously_known, **extraction.extracted_fields}
        newly_learned = {k: v for k, v in extraction.extracted_fields.items() if k not in previously_known}

        result = validation.validate_patient_fields(merged)
        missing = validation.missing_required_fields(result.normalized_fields, required_fields)

        return {
            "extracted_fields": merged,
            "newly_learned_fields": newly_learned,
            "normalized_fields": result.normalized_fields,
            "validation_errors": result.errors,
            "missing_fields": missing,
        }

    def clarify_node(state: AgentState) -> dict:
        errors_text = "; ".join(state["validation_errors"])
        response = (
            f"I need to double-check something: {errors_text}. Could you clarify that?"
        )
        return {"turn_response": response, "done": False}

    def followup_node(state: AgentState) -> dict:
        count = state.get("followup_count", 0) + 1
        target_field = patient_intake.select_next_missing_field(state["missing_fields"])
        question = patient_intake.generate_followup_question(
            client, target_field, newly_learned_fields=state.get("newly_learned_fields")
        )
        return {"turn_response": question, "followup_count": count, "done": False}

    def insufficient_info_node(state: AgentState) -> dict:
        response = (
            "I still don't have enough information to make an assessment after "
            f"several questions (still missing: {', '.join(state['missing_fields'])}). "
            "Please provide these directly, or consult a healthcare professional in person."
        )
        return {"turn_response": response, "done": False}

    def predict_node(state: AgentState) -> dict:
        result = disease_prediction.predict(state["normalized_fields"])
        return {
            "prediction_probability": result.probability,
            "prediction_class": result.predicted_class,
            "prediction_result": result,
        }

    def explain_node(state: AgentState) -> dict:
        prediction = state["prediction_result"]
        contributions = risk_explanation.get_shap_contributions(prediction)
        narrative = risk_explanation.build_narrative(client, prediction, contributions)
        return {"shap_contributions": contributions, "narrative": narrative}

    def retrieve_node(state: AgentState) -> dict:
        outcome = knowledge_retrieval.retrieve_evidence(state["shap_contributions"])
        return {"retrieval_query": outcome.query, "retrieved_passages": outcome.passages}

    def synthesize_node(state: AgentState) -> dict:
        parts = [state["narrative"]]
        if state["retrieved_passages"]:
            parts.append("\nSupporting evidence:")
            for p in state["retrieved_passages"]:
                parts.append(f"- {p.text[:200]}... ({p.citation()})")
        else:
            parts.append("\n(No directly relevant passage found in the knowledge base for this finding.)")
        parts.append(f"\n{DISCLAIMER}")
        return {"turn_response": "\n".join(parts), "done": True}

    def route_after_intake(state: AgentState) -> str:
        if state["validation_errors"]:
            return "clarify"
        if state["missing_fields"]:
            if state.get("followup_count", 0) >= max_followups:
                return "insufficient_info"
            return "followup"
        return "predict"

    graph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("followup", followup_node)
    graph.add_node("insufficient_info", insufficient_info_node)
    graph.add_node("predict", predict_node)
    graph.add_node("explain", explain_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "clarify": "clarify",
            "followup": "followup",
            "insufficient_info": "insufficient_info",
            "predict": "predict",
        },
    )
    graph.add_edge("clarify", END)
    graph.add_edge("followup", END)
    graph.add_edge("insufficient_info", END)
    graph.add_edge("predict", "explain")
    graph.add_edge("explain", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)


    return graph.compile()
