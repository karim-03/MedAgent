"""
client.py
Thin, config-driven wrapper around the Ollama Python client.

Design choices (see docs/architecture.md Section 5.1-5.3):
- Model name is never hardcoded — read from config/settings.yaml, so
  swapping Qwen2.5-7B for the Gemma 3 4B "lite mode" fallback is a config
  change, not a code change.
- This module ONLY does LLM plumbing (connect, generate, chat, structured
  JSON output, automatic fallback). It contains no clinical logic and no
  prompt content — those belong to the agent layer (P4), which will import
  this client rather than talk to Ollama directly. Keeping the boundary
  here means the agent layer is swappable to a different LLM runtime later
  without the agent's own code changing.
- Errors are translated into specific, actionable messages (server not
  running vs. model not pulled vs. genuine failure) rather than a raw
  connection traceback, because "the LLM didn't respond" needs a different
  fix in each of those three cases.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ollama
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/settings.yaml")


def _load_llm_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")
    settings = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return settings["llm"]


@dataclass
class GenerationResult:
    text: str
    model_used: str
    used_fallback: bool
    prompt_tokens: int
    completion_tokens: int
    total_duration_s: float
    tokens_per_second: float


class OllamaNotRunningError(RuntimeError):
    """Raised when the Ollama server itself isn't reachable."""


class ModelNotPulledError(RuntimeError):
    """Raised when Ollama is running but the configured model isn't pulled."""


class LocalLLMClient:
    def __init__(self, host: str = "http://localhost:11434", config: Optional[dict] = None):
        self.host = host
        self.config = config or _load_llm_config()
        self.primary_model = self.config["model"]
        self.fallback_model = self.config.get("fallback_model")
        self._client = ollama.Client(host=host)

    def health_check(self) -> list[str]:
        """Returns the list of locally available model names. Raises
        OllamaNotRunningError if the server itself can't be reached —
        this check should run before anything else in P3/P4, so failures
        are diagnosed at the right layer."""
        try:
            response = self._client.list()
        except Exception as exc:
            raise OllamaNotRunningError(
                "Could not reach the Ollama server at "
                f"{self.host}. Is it running? Start it with `ollama serve` "
                "(or check the Ollama desktop app is open)."
            ) from exc

        # The ollama client library has changed its return shape across
        # versions (plain dict vs. a typed ListResponse object) — handle
        # both rather than pinning to one and breaking on an upgrade.
        models = response["models"] if isinstance(response, dict) else response.models
        return [m["model"] if isinstance(m, dict) else m.model for m in models]

    def _ensure_model_available(self, model_name: str) -> None:
        available = self.health_check()
        if not any(model_name in m for m in available):
            raise ModelNotPulledError(
                f"Model '{model_name}' is not pulled. Run: ollama pull {model_name}"
            )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_format: bool = False,
        options: Optional[dict] = None,
    ) -> GenerationResult:
        """Generate a completion, automatically falling back to
        fallback_model if the primary model is unavailable or fails for
        any reason — not pulled, out-of-memory, or any other runtime
        error. If the fallback attempt ALSO fails, that final error is
        what propagates (not the original one), since it's the more
        actionable failure: at that point the fix is "pull at least one
        of these two models" or "start Ollama", not a code-level issue."""
        last_exc = None
        for model_name, is_fallback in self._model_attempt_order():
            try:
                self._ensure_model_available(model_name)
                return self._generate_with_model(model_name, prompt, system, json_format, options, is_fallback)
            except Exception as exc:
                last_exc = exc
                if not is_fallback and self.fallback_model:
                    logger.warning(
                        "Primary model '%s' unavailable/failed (%s). Retrying with fallback '%s'.",
                        model_name, exc, self.fallback_model,
                    )
                continue
        raise last_exc or RuntimeError("No model succeeded and no fallback is configured.")

    def _model_attempt_order(self):
        yield self.primary_model, False
        if self.fallback_model:
            yield self.fallback_model, True

    def _generate_with_model(self, model_name, prompt, system, json_format, options, is_fallback) -> GenerationResult:
        start = time.perf_counter()
        response = self._client.generate(
            model=model_name,
            prompt=prompt,
            system=system,
            format="json" if json_format else None,
            options=options or {},
        )
        elapsed = time.perf_counter() - start

        # Ollama reports its own eval counts/durations (nanoseconds) — more
        # accurate than wall-clock Python timing for tokens/sec, since it
        # excludes Python-side overhead.
        eval_count = getattr(response, "eval_count", None) or response.get("eval_count", 0)
        eval_duration_ns = getattr(response, "eval_duration", None) or response.get("eval_duration", 0)
        prompt_eval_count = getattr(response, "prompt_eval_count", None) or response.get("prompt_eval_count", 0)
        text = getattr(response, "response", None) or response.get("response", "")

        tokens_per_second = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else 0.0

        return GenerationResult(
            text=text,
            model_used=model_name,
            used_fallback=is_fallback,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_duration_s=elapsed,
            tokens_per_second=tokens_per_second,
        )
