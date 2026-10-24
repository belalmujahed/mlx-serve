"""Model catalog: downloaded models, curated picks and recent HF discoveries."""

import logging
import re
import threading
import time
from typing import Dict, List, Optional

from huggingface_hub import HfApi, scan_cache_dir

from .config import config

log = logging.getLogger("proxy")

_EXCLUDE = re.compile(r"(?i)vl|vision|embed|whisper|tts|audio|image|lora|draft")
_INCLUDE = re.compile(r"(?i)instruct|coder|thinking|-it-|glm|gpt-oss|deepseek")
_DISCOVERY_TTL = 3600

_discovered: List[str] = []
_discovered_at = 0.0


def downloaded_models() -> List[str]:
    """Repo ids of MLX models already present in the local HF cache."""
    try:
        repos = scan_cache_dir().repos
    except Exception:
        return []
    return sorted(
        r.repo_id
        for r in repos
        if r.repo_type == "model" and r.repo_id.startswith("mlx-community/")
    )


def refresh_discovery() -> None:
    """Fetch recent mlx-community chat models from HF into the cache (network)."""
    global _discovered, _discovered_at
    if not config.DISCOVER_MODELS:
        return
    try:
        found = HfApi().list_models(
            author="mlx-community",
            filter="text-generation",
            sort="last_modified",
            limit=300,
        )
        ids = [
            m.id for m in found if _INCLUDE.search(m.id) and not _EXCLUDE.search(m.id)
        ]
        _discovered = ids[: config.DISCOVER_LIMIT]
        log.info("Discovered %d recent mlx-community models", len(_discovered))
    except Exception as e:
        log.warning("Model discovery failed: %s", e)
    _discovered_at = time.time()


def discover_models() -> List[str]:
    """Cached discovery results; never blocks on the network.

    Claude Code gives /v1/models only a few seconds, so stale data is served
    while a background thread refreshes it.
    """
    global _discovered_at
    if time.time() - _discovered_at >= _DISCOVERY_TTL:
        _discovered_at = time.time()  # avoid concurrent refreshes
        threading.Thread(target=refresh_discovery, daemon=True).start()
    return _discovered


def list_catalog(current: Optional[str]) -> List[Dict[str, object]]:
    """Ordered, de-duplicated models: current, downloaded, curated, discovered."""
    local = set(downloaded_models())
    ordered = [current] if current else []
    ordered += sorted(local) + config.CURATED_MODELS + discover_models()
    entries = []
    for model_id in dict.fromkeys(ordered):
        state = (
            "loaded"
            if model_id == current
            else ("downloaded" if model_id in local else "will download")
        )
        entries.append({"id": model_id, "state": state})
    return entries


def is_model_repo(name: str) -> bool:
    """HF repo ids look like 'org/name'; Claude model aliases never contain '/'."""
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", name))
