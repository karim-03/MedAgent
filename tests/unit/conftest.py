"""
Shared pytest fixtures for tests/unit/.

COMPLETE_PATIENT was independently hand-typed in three places across two
test files (test_tools.py had it twice under two different names,
test_agent_graph.py had its own copy) — consolidated here so there's one
canonical "valid, complete patient" test case instead of three copies that
could silently drift out of sync with each other.
"""

# A complete, valid patient covering every required field — used across
# validation, prediction, SHAP/narrative grounding, and agent graph tests.
# thal=3 (reversible defect) is deliberately the profile used for the
# grounding regression tests (docs/agent_core_findings.md run #3) — this
# is the SAME patient case that was verified against real hardware output
# throughout P4, not an arbitrary test fixture.
COMPLETE_PATIENT = {
    "age": 58, "sex": "man", "cp": 3, "trestbps": 145, "chol": 260,
    "fbs": "no", "restecg": 0, "thalach": 132, "exang": "yes",
    "oldpeak": 2.1, "slope": 1, "ca": 2, "thal": 3,
}
