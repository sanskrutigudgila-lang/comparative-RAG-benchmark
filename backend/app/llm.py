"""LLM access: Claude via langchain-anthropic, with an on-disk prompt-hash cache.

If no ANTHROPIC_API_KEY is set (or RAG_OFFLINE=1) the callers use their own deterministic
heuristics instead; `config.OFFLINE` tells them which mode is active, and results are labelled.
The key is only ever read from the environment.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _client():
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=config.LLM_MODEL, temperature=0, max_tokens=2048)


def complete(system: str, user: str, *, cache: bool = True) -> str:
    if config.OFFLINE:
        raise RuntimeError("LLM called in offline mode")
    key = hashlib.sha256(json.dumps([config.LLM_MODEL, system, user]).encode()).hexdigest()
    path = config.CACHE_DIR / f"{key}.txt"
    if cache and path.exists():
        return path.read_text()
    out = _client().invoke([("system", system), ("human", user)]).content
    text = out if isinstance(out, str) else "".join(b.get("text", "") for b in out if isinstance(b, dict))
    if cache:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text


def complete_json(system: str, user: str):
    raw = complete(system, user + "\n\nReturn ONLY valid JSON, no prose.")
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)
