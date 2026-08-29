"""
config/loader.py
Single shared loader for config/settings.yaml.

Previously, llm/client.py, tools/disease_prediction.py, and agent/graph.py
each independently defined their own `CONFIG_PATH = Path("config/settings.yaml")`
constant and repeated the same read-and-parse call — the same magic path
and the same three lines of logic, three times, with no caching (each
loader re-read and re-parsed the file from disk on every call). In
practice each was only ever called once per process, so this was never a
real performance problem — but it was real, findable duplication with no
upside, which is what this module removes: one path, one parse, cached.
"""

from pathlib import Path
from typing import Optional

import yaml

CONFIG_PATH = Path("config/settings.yaml")

_settings_cache: Optional[dict] = None


def load_settings(force_reload: bool = False) -> dict:
    """Returns the full parsed settings.yaml, cached after the first call.
    Pass force_reload=True to bypass the cache (e.g. a test that swaps in
    a different config file mid-run)."""
    global _settings_cache
    if _settings_cache is None or force_reload:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")
        _settings_cache = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return _settings_cache
