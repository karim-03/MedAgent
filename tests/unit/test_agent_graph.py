"""
Unit tests for agent/graph.py — control flow only (routing between intake,
clarify, followup, insufficient_info, predict->explain->retrieve->
synthesize). These use a scripted FakeClient rather than real Ollama, so
they test exactly what matters most at this layer: does the graph route
correctly given a certain LLM output, not whether the LLM's output is
good (that's what tests/unit/test_llm_client.py's real-Ollama tests and
the P4 driver script on real hardware are for).

rag.retrieve.retrieve() is monkeypatched too — it needs the real embedding
model, which is a P2-covered, separately-tested concern; re-testing it
here would just be re-testing P2, not P4's actual new logic.

Run with: pytest tests/unit/test_agent_graph.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.graph import build_graph
from llm.client import LocalLLMClient, GenerationResult
from tools.knowledge_retrieval import RetrievalOutcome

COMPLETE_PATIENT = {
    "age": 58, "sex": "male", "cp": 3, "trestbps": 145, "chol": 260,
    "fbs": "no", "restecg": 0, "thalach": 132, "exang": "yes",
    "oldpeak": 2.1, "slope": 1, "ca": 2, "thal": 3,
}


class ScriptedClient(LocalLLMClient):
    """A fake client whose generate() return value is set per-test rather
    than driven by a real model — makes graph routing deterministic."""

    def __init__(self):
        self.next_extraction: dict = {}
        self.next_followup_text = "What is your cholesterol level?"
        self.next_narrative_text = "Fake narrative. Probability 78%. Reversible defect."
        self.calls = []

    def generate(self, prompt, system=None, json_format=False, options=None):
        self.calls.append({"prompt": prompt, "system": system, "json_format": json_format})
        if json_format:
            text = json.dumps(self.next_extraction)
        elif system and "exactly one field" in system.lower():
            text = self.next_followup_text
        else:
            text = self.next_narrative_text
        return GenerationResult(
            text=text, model_used="scripted", used_fallback=False,
            prompt_tokens=1, completion_tokens=1, total_duration_s=0.01, tokens_per_second=1.0,
        )


@pytest.fixture
def no_op_retrieval(monkeypatch):
    """Every graph test that reaches the predict->synthesize path needs
    retrieval mocked out — this fixture does it once so tests don't repeat
    the monkeypatch."""
    def fake_retrieve_evidence(shap_contributions, k=2):
        return RetrievalOutcome(query="fake query", passages=[])
    monkeypatch.setattr("agent.graph.knowledge_retrieval.retrieve_evidence", fake_retrieve_evidence)


def _initial_state(message: str) -> dict:
    return {"messages": [{"role": "user", "content": message}], "followup_count": 0}


def test_missing_fields_routes_to_followup_not_predict():
    client = ScriptedClient()
    client.next_extraction = {"age": 58, "sex": "male"}  # far from complete
    graph = build_graph(client)

    result = graph.invoke(_initial_state("I'm a 58 year old man."))

    assert result["done"] is False
    assert result["followup_count"] == 1
    assert "prediction_probability" not in result or result.get("prediction_probability") is None
    assert result["turn_response"] == client.next_followup_text


def test_invalid_field_routes_to_clarify_not_followup():
    client = ScriptedClient()
    client.next_extraction = {**COMPLETE_PATIENT, "trestbps": 9999}  # out of plausible range
    graph = build_graph(client)

    result = graph.invoke(_initial_state("here's my info"))

    assert result["done"] is False
    assert "outside the plausible range" in result["turn_response"]
    # clarify must not consume a followup — it's a correction, not a new question
    assert result.get("followup_count", 0) == 0


def test_complete_valid_fields_runs_full_pipeline_to_synthesize(no_op_retrieval):
    client = ScriptedClient()
    client.next_extraction = COMPLETE_PATIENT
    graph = build_graph(client)

    result = graph.invoke(_initial_state("here's everything"))

    assert result["done"] is True
    assert result["prediction_probability"] is not None
    assert result["shap_contributions"]
    assert client.next_narrative_text in result["turn_response"]
    assert "educational capstone project" in result["turn_response"]  # disclaimer present


def test_followup_safety_valve_triggers_insufficient_info():
    client = ScriptedClient()
    client.next_extraction = {"age": 58}  # perpetually incomplete
    graph = build_graph(client)

    state = _initial_state("I'm 58")
    state["followup_count"] = 8  # already at the configured max

    result = graph.invoke(state)

    assert result["done"] is False
    assert "still don't have enough information" in result["turn_response"]


def test_sex_as_man_reaches_prediction_without_validation_error(no_op_retrieval):
    """Regression test for the exact P3 finding: 'man' must not be
    treated as an invalid/missing sex value."""
    client = ScriptedClient()
    client.next_extraction = {**COMPLETE_PATIENT, "sex": "man"}
    graph = build_graph(client)

    result = graph.invoke(_initial_state("here's my info, I'm a man"))

    assert result["validation_errors"] == []
    assert result["normalized_fields"]["sex"] == 1
    assert result["done"] is True


def test_followup_targets_highest_priority_missing_field_deterministically():
    """Regression test for the repeated-question issue found by actually
    running the agent (docs/agent_core_findings.md): field selection must
    not be left to the LLM's independent judgment each turn — it should
    deterministically pick the same highest-priority field given the same
    missing set, every time."""
    client = ScriptedClient()
    # thal/ca/oldpeak/exang/thalach/slope/cp all missing — thal is highest priority
    client.next_extraction = {"age": 58, "sex": "male", "trestbps": 145, "chol": 260, "fbs": "no", "restecg": 0}
    graph = build_graph(client)

    graph.invoke(_initial_state("partial info"))

    followup_call = next(c for c in client.calls if c["system"] and "exactly one field" in c["system"].lower())
    assert "thal" in followup_call["prompt"].lower()


def test_followup_prompt_acknowledges_newly_learned_fields_on_second_turn(no_op_retrieval):
    """Turn 2 should tell the LLM what was just learned, so the question
    doesn't read as if the agent ignored the patient's last message —
    this is the actual fix for the observed back-to-back-identical-question
    behavior."""
    client = ScriptedClient()
    client.next_extraction = {"chol": 260}  # only chol learned this turn
    graph = build_graph(client)

    state = _initial_state("my cholesterol was 260")
    state["extracted_fields"] = {"age": 58, "sex": "male"}  # already known from a prior turn
    state["followup_count"] = 1

    graph.invoke(state)

    followup_call = next(c for c in client.calls if c["system"] and "exactly one field" in c["system"].lower())
    assert "chol=260" in followup_call["prompt"]
