"""Step 4: dense 1024-dim bge-large-en-v1.5 embeddings. RAG_EMBED_MODEL=hash selects a tiny
deterministic hashing embedder for unit tests / no-download runs (retrieval quality is lower)."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings

from .. import config


class HashEmbeddings(Embeddings):
    dim = 512

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        toks = re.findall(r"[a-z0-9]+", text.lower())
        for t in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1 if (h >> 64) % 2 else -1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts): return [self._vec(t) for t in texts]
    def embed_query(self, text): return self._vec(text)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    if config.EMBED_MODEL == "hash":
        return HashEmbeddings()
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=config.EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
