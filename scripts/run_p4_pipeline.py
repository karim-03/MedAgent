"""
One-shot driver for P4: run the actual LangGraph agent through a
multi-turn simulated conversation, using the real local LLM (not the
ScriptedClient the unit tests use). This is the first point in the project
where you can watch the whole loop — intake, follow-up question, another
intake turn, prediction, SHAP explanation, evidence retrieval, synthesis —
happen for real, end to end.

Needs: Ollama running with the configured model pulled (P3), and the FAISS
index built (P2 — run `python scripts/run_p2_pipeline.py` first if you
haven't since embeddings/faiss_index/ isn't committed to the repo).


Usage:
    python scripts/run_p4_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from agent.graph import build_graph
from llm.client import LocalLLMClient, OllamaNotRunningError

# A deliberately incomplete-then-complete conversation, so the follow-up
# loop actually exercises for real, not just the single-shot happy path.
CONVERSATION_TURNS = [
    "I'm a 58 year old man with chest tightness when I climb stairs. "
    "My resting blood pressure was 145 last checkup.",
    "My cholesterol was 260. My fasting blood sugar was checked recently "
    "and it was NOT over 120, so that's a no.",
    "My resting ECG was normal. During a stress test my max heart rate was 132, "
    "and I did get chest pain during the test — the ST depression reading was 2.1, "
    "described as flat slope. The angiogram showed 2 blocked vessels, and the "
    "thallium test showed a reversible defect.",
]


def main():
    client = LocalLLMClient()
    try:
        client.health_check()
    except OllamaNotRunningError as exc:
        print(f"\n{exc}\n")
        sys.exit(1)

    graph = build_graph(client)
    state = {"messages": [], "followup_count": 0}

    for turn_num, user_message in enumerate(CONVERSATION_TURNS, start=1):
        print(f"\n{'='*70}\nTurn {turn_num}")
        print(f"Patient: {user_message}")

        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_message}]
        state = graph.invoke(state)

        print(f"\nAgent: {state['turn_response']}")

        if state.get("done"):
            print(f"\n{'='*70}")
            print("Conversation complete.")
            print(f"Final prediction: class={state['prediction_class']} "
                  f"probability={state['prediction_probability']:.3f}")
            print(f"SHAP contributions: {state['shap_contributions']}")
            print(f"Retrieval query used: {state['retrieval_query']}")
            break
    else:
        print("\nConversation ended without reaching a prediction — check the "
              "follow-up questions above; the scripted patient messages may not "
              "have covered every field the LLM asked about.")


if __name__ == "__main__":
    main()
