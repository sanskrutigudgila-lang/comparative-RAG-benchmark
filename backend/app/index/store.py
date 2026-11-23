"""Step 5(a)/6: the dual index (Chroma HNSW vectors + BM25 + knowledge graph) behind one object."""
from __future__ import annotations

import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .. import config
from ..ingest.chunking import split_documents
from ..ingest.loaders import load_file
from .embeddings import get_embeddings
from .graph import KnowledgeGraph


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class IndexStore:
    def __init__(self, hp: config.Hyperparameters | None = None, collection: str = "rag"):
        self.hp = hp or config.Hyperparameters()
        self.chunks: dict[str, Document] = {}
        self.kg = KnowledgeGraph()
        self.vs = Chroma(collection_name=collection, embedding_function=get_embeddings(),
                         collection_metadata={"hnsw:space": "cosine"})
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self.ingest_log: list[dict] = []

    def ingest_paths(self, paths) -> int:
        new: list[Document] = []
        for p in paths:
            if Path(p).name.lower().startswith("gold_qa"):   # answer key must never enter the index (benchmark leakage)
                continue
            new += split_documents(load_file(p), self.hp)
        new = [c for c in new if c.metadata["chunk_id"] not in self.chunks]
        if not new:
            return 0
        self.vs.add_documents(new, ids=[c.metadata["chunk_id"] for c in new])
        for c in new:
            self.chunks[c.metadata["chunk_id"]] = c
        self.kg.build(new)
        self.kg.train_node2vec()
        self._bm25_ids = list(self.chunks)
        self._bm25 = BM25Okapi([tokenize(self.chunks[i].page_content) for i in self._bm25_ids])
        self.ingest_log.append({"files": [str(p) for p in paths], "chunks": len(new)})
        return len(new)

    # --- three ranked lists used by the pipelines -------------------------------
    def vector_search(self, query: str, k: int, practice_area: str | None = None) -> list[tuple[str, float]]:
        res = self.vs.similarity_search_with_score(query, k=k)
        return [(d.metadata["chunk_id"], 1 - s) for d, s in res]

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self._bm25_ids[i], float(scores[i])) for i in order if scores[i] > 0]
