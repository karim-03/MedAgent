"""
One-shot driver for P3: verify Ollama is running, the configured model is
pulled, and benchmark latency/throughput on prompts representative of what
the agent will actually need in P4 (structured field extraction, follow-up
question generation, short narrative synthesis).

**This script must be run on your actual machine (RTX 4060) — it cannot be
run inside Claude's sandboxed dev environment, which has no GPU and no
access to Ollama's model registry.** That's a genuine limitation of the
tool sandbox, not a stand-in for real verification: run this, then share
the printed output (or the saved JSON/markdown files) so the architecture
doc's VRAM/latency assumptions can be checked against real numbers instead
of estimates.

Usage:
    ollama serve                                    # if not already running
    ollama pull qwen2.5:7b-instruct-q4_K_M           # one-time download
    ollama pull gemma3:4b                            # fallback model, also one-time
    python scripts/run_p3_pipeline.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subprocess

from llm.client import LocalLLMClient, OllamaNotRunningError

OUTPUT_MD = Path("docs/llm_verification_findings.md")
OUTPUT_JSON = Path("outputs/llm_benchmark_results.json")

# Representative of the three LLM-only jobs the agent will actually do in
# P4 — never disease prediction, that stays with the ML model.
BENCHMARK_PROMPTS = [
    {
        "name": "structured_field_extraction",
        "system": (
            "Extract patient information into JSON. Only include fields "
            "explicitly stated. Do not guess missing values."
        ),
        "prompt": (
            "Patient message: 'I'm a 58 year old man, my resting blood "
            "pressure was 145 last checkup and I get chest tightness "
            "when I climb stairs.'\n"
            "Extract into JSON with keys: age, sex, resting_bp, symptoms."
        ),
        "json_format": True,
    },
    {
        "name": "follow_up_question_generation",
        "system": "You are a clinical intake assistant. Ask exactly ONE concise follow-up question.",
        "prompt": (
            "Known so far: age=58, sex=male, resting_bp=145, "
            "symptom=exertional chest tightness. Still missing: cholesterol "
            "level, whether a stress test or ECG has been done, and family "
            "history. Ask the single most useful next question."
        ),
        "json_format": False,
    },
    {
        "name": "risk_narrative_synthesis",
        "system": (
            "You are summarizing an ML model's output for a clinician. "
            "State the numbers as given — never invent or adjust them."
        ),
        "prompt": (
            "Model prediction: heart disease present, probability 0.78. "
            "Top contributing factors (SHAP): thal (reversible defect), "
            "ca=2 (vessels blocked), oldpeak=2.1 (ST depression). "
            "Write a 3-sentence plain-language summary for a clinician."
        ),
        "json_format": False,
    },
]


def get_gpu_memory_report() -> str:
    try:
        result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or "(ollama ps returned no output — is a model currently loaded?)"
    except FileNotFoundError:
        return "(could not run `ollama ps` — is the ollama CLI on your PATH?)"


def main():
    client = LocalLLMClient()

    print(f"Checking Ollama server at {client.host} ...")
    try:
        available = client.health_check()
    except OllamaNotRunningError as exc:
        print(f"\n{exc}\n")
        sys.exit(1)
    print(f"Server reachable. Locally available models: {available}\n")

    for model_name in [client.primary_model, client.fallback_model]:
        if model_name and not any(model_name in m for m in available):
            print(f"WARNING: '{model_name}' is not pulled yet. Run: ollama pull {model_name}")
    print()

    results = []
    for case in BENCHMARK_PROMPTS:
        print(f"Running: {case['name']} ...")
        try:
            result = client.generate(
                prompt=case["prompt"], system=case["system"], json_format=case["json_format"]
            )
            print(f"  model={result.model_used} fallback={result.used_fallback} "
                  f"time={result.total_duration_s:.2f}s "
                  f"tokens/s={result.tokens_per_second:.1f} "
                  f"completion_tokens={result.completion_tokens}")
            print(f"  output: {result.text[:150]!r}")
            results.append({"case": case["name"], **result.__dict__})
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append({"case": case["name"], "error": str(exc)})
        print()

    gpu_report = get_gpu_memory_report()
    print("`ollama ps` output (check this against the ~4.5GB VRAM budget in docs/architecture.md Section 9):")
    print(gpu_report)

    Path("outputs").mkdir(exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps({"results": results, "ollama_ps": gpu_report}, indent=2), encoding="utf-8")
    print(f"\nSaved raw results to {OUTPUT_JSON}")
    print("Share this file's contents (or just paste the console output above) "
          "so the real numbers can go into docs/llm_verification_findings.md.")



if __name__ == "__main__":
    main()
