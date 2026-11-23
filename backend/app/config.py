"""Central configuration: the 4 spec hyperparameters, model IDs and paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
EVAL_DIR = DATA_DIR / "eval"          # gold questions/answers live OUTSIDE the corpus so they can never be indexed
STORE_DIR = DATA_DIR / "store"
CACHE_DIR = DATA_DIR / "llm_cache"

LLM_MODEL = os.getenv("RAG_LLM_MODEL", "claude-sonnet-5-5")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-large-en-v1.5")  # 1024-dim
# "offline" = deterministic heuristics, no API key needed. Results are labelled as such.
OFFLINE = os.getenv("RAG_OFFLINE", "").lower() in {"1", "true", "yes"} or not os.getenv("ANTHROPIC_API_KEY")


@dataclass
class Hyperparameters:
    chunk_size: int = 600        # tokens
    chunk_overlap: int = 60      # tokens
    hyde_passages: int = 3       # N
    hop_radius: int = 2          # k
    rrf_alpha: int = 60          # RRF constant
    top_k: int = 3               # chunks passed to synthesis

    def as_dict(self) -> dict:
        return asdict(self)


PRACTICE_AREAS = {
    "tax": "Tax & Regulatory Services",
    "grc": "Risk Advisory & Governance (GRC)",
    "deals": "Deal Advisory & Corporate Finance",
}
ENGINES = ("standard", "hyde", "graph")
