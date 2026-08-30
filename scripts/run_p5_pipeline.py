"""
One-shot driver for P5: run the same representative conversation as P4,
then save and display the actual structured report — the real
downloadable artifact this milestone adds. P4's driver script is left
alone (already verified across 6 real hardware runs); this one adds the
save-to-file step and prints the full report so it's easy to see exactly
what a patient/clinician would receive.

Needs: Ollama running with the configured model pulled (P3), and the FAISS
index built (P2).

Usage:
    python scripts/run_p5_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from agent.graph import build_graph
from llm.client import LocalLLMClient, OllamaNotRunningError

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

    for user_message in CONVERSATION_TURNS:
        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_message}]
        state = graph.invoke(state)
        if state.get("done"):
            break

    if not state.get("done"):
        print("Conversation did not reach a completed report — check the "
              "follow-up questions; the scripted messages may not cover "
              "everything the LLM asked about this run.")
        sys.exit(1)

    print("=" * 70)
    print("Chat summary (what the agent said in the conversation):")
    print("=" * 70)
    print(state["turn_response"])

    from tools.report_generator import save_report_markdown
    saved_path = save_report_markdown(state["report"])

    print("\n" + "=" * 70)
    print(f"Full report saved to: {saved_path}")
    print("=" * 70)
    print(state["report_markdown"])


if __name__ == "__main__":
    main()
