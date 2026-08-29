"""
Unit tests for llm/client.py

Split deliberately into two groups:
1. Logic tests (config loading, fallback ordering, fallback control flow)
   — these run anywhere, no Ollama server needed, using a real
   LocalLLMClient with its network calls monkeypatched.
2. Integration tests — need a real, reachable Ollama server with the
   configured model actually pulled. Skipped (not failed) if unreachable,
   for the same reason as the embedding-model tests in test_rag.py: that's
   an environment property, not a code bug. On your machine, with Ollama
   running and the model pulled, these execute for real.

Run with: pytest tests/unit/test_llm_client.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm.client import LocalLLMClient, OllamaNotRunningError, GenerationResult


def _ollama_server_available() -> bool:
    try:
        LocalLLMClient().health_check()
        return True
    except OllamaNotRunningError:
        return False


OLLAMA_AVAILABLE = _ollama_server_available()
skip_if_no_ollama = pytest.mark.skipif(
    not OLLAMA_AVAILABLE, reason="No reachable Ollama server in this environment"
)


# ---------- logic tests (no server needed) ----------

def test_config_defines_primary_and_fallback_model():
    client = LocalLLMClient()
    assert client.primary_model
    assert client.fallback_model  # architecture doc requires a configured fallback


def test_health_check_raises_clear_error_against_unreachable_host():
    # Deliberately point at a port nothing listens on, regardless of
    # whether a real Ollama happens to be running elsewhere in this
    # environment — makes this test deterministic either way.
    client = LocalLLMClient(host="http://localhost:1")
    with pytest.raises(OllamaNotRunningError):
        client.health_check()


def test_model_attempt_order_yields_primary_then_fallback():
    client = LocalLLMClient()
    order = list(client._model_attempt_order())
    assert order[0] == (client.primary_model, False)
    assert order[1] == (client.fallback_model, True)


def test_generate_falls_back_when_primary_raises(monkeypatch):
    client = LocalLLMClient()
    monkeypatch.setattr(client, "_ensure_model_available", lambda name: None)

    call_log = []

    def fake_generate_with_model(model_name, prompt, system, json_format, options, is_fallback):
        call_log.append(model_name)
        if not is_fallback:
            raise RuntimeError("simulated OOM")
        return GenerationResult(
            text="ok", model_used=model_name, used_fallback=True,
            prompt_tokens=5, completion_tokens=5, total_duration_s=0.1, tokens_per_second=50.0,
        )

    monkeypatch.setattr(client, "_generate_with_model", fake_generate_with_model)
    result = client.generate("test prompt")

    assert call_log == [client.primary_model, client.fallback_model]
    assert result.used_fallback is True
    assert result.text == "ok"


def test_generate_raises_final_error_when_both_models_fail(monkeypatch):
    client = LocalLLMClient()
    monkeypatch.setattr(client, "_ensure_model_available", lambda name: None)

    def always_fails(model_name, prompt, system, json_format, options, is_fallback):
        raise RuntimeError(f"failure for {model_name}")

    monkeypatch.setattr(client, "_generate_with_model", always_fails)
    with pytest.raises(RuntimeError, match=client.fallback_model):
        client.generate("test prompt")


def test_generate_does_not_use_fallback_when_primary_succeeds(monkeypatch):
    client = LocalLLMClient()
    monkeypatch.setattr(client, "_ensure_model_available", lambda name: None)

    call_log = []

    def fake_generate_with_model(model_name, prompt, system, json_format, options, is_fallback):
        call_log.append(model_name)
        return GenerationResult(
            text="ok", model_used=model_name, used_fallback=is_fallback,
            prompt_tokens=5, completion_tokens=5, total_duration_s=0.1, tokens_per_second=50.0,
        )

    monkeypatch.setattr(client, "_generate_with_model", fake_generate_with_model)
    result = client.generate("test prompt")

    assert call_log == [client.primary_model]
    assert result.used_fallback is False


# ---------- integration tests (need a real, reachable Ollama server) ----------

@skip_if_no_ollama
def test_real_health_check_lists_models():
    client = LocalLLMClient()
    models = client.health_check()
    assert isinstance(models, list)


@skip_if_no_ollama
def test_real_generate_returns_nonempty_text():
    client = LocalLLMClient()
    result = client.generate("Reply with exactly the word: OK")
    assert result.text.strip()
    assert result.total_duration_s > 0


@skip_if_no_ollama
def test_real_generate_json_format_is_valid_json():
    import json
    client = LocalLLMClient()
    result = client.generate(
        "Return a JSON object with a single key 'status' set to 'ok'.",
        json_format=True,
    )
    parsed = json.loads(result.text)
    assert "status" in parsed
