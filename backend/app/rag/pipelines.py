"""The three engines as LangGraph StateGraphs sharing one RAGState.

Toggle model (Tab 3): `disabled_nodes` -> the node becomes a pass-through; `disabled_edges` -> the
edge is omitted when the graph is built and the upstream node routes to END instead.
Graph RAG contains a real cycle: rerank -> query_expansion when retrieved context is thin.
"""
from __future__ import annotations

import time
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .. import config, llm
from ..index.store import IndexStore
from ..prompts import HYDE_ANGLES
from .fusion import rerank, rrf
from .synthesis import synthesize

# Offline stand-in for LLM query expansion / HyDE vocabulary bridging.
LEXICON = {
    "fine": "penalty", "fines": "penalty", "walk away": "terminate", "acquirer": "Buyer Orion Holdings",
    "central bank": "RBI Reserve Bank", "tax authority": "CBDT Income Tax Department", "onboarding": "Customer Due Diligence",
    "lender": "Regulated Entities", "earnings": "EBITDA", "outsourcing": "contract services", "r&d": "research",
    "local subsidiaries": "Indian subsidiaries", "parent": "head office Seller Zenith Pharma Ltd",
    "regulator": "regulatory action", "breach": "contravention", "checks": "due diligence",
}


class RAGState(TypedDict, total=False):
    query: str
    practice_area: str
    queries: list[str]
    candidates: dict[str, list[str]]     # source name -> ranked chunk ids
    fused: list[tuple[str, float]]
    context_ids: list[str]
    answer: str
    loops: int
    timings: dict[str, float]            # ms, by category: retrieval / graph / llm / other
    trace: list[str]


EDGES = {
    "standard": [("retrieve", "synthesize")],
    "hyde": [("hyde_generate", "retrieve"), ("retrieve", "synthesize")],
    "graph": [("query_expansion", "hybrid_search"), ("hybrid_search", "rrf_fusion"),
              ("rrf_fusion", "rerank"), ("rerank", "synthesis")],
}
ENTRY = {"standard": "retrieve", "hyde": "hyde_generate", "graph": "query_expansion"}
LABELS = {"query_expansion": "Query Expansion", "hybrid_search": "Hybrid Vector/Graph Search",
          "rrf_fusion": "RRF Fusion", "rerank": "Rerank", "synthesis": "Synthesis", "retrieve": "Vector Retrieve",
          "hyde_generate": "HyDE Passage Generation", "synthesize": "Synthesis"}
CATEGORY = {"query_expansion": "llm", "hyde_generate": "llm", "hybrid_search": "retrieval", "retrieve": "retrieval",
            "rrf_fusion": "other", "rerank": "other", "synthesis": "llm", "synthesize": "llm"}


def workflow_description(engine: str) -> dict:
    nodes = [{"id": ENTRY[engine], "label": LABELS[ENTRY[engine]]}]
    for a, b in EDGES[engine]:
        nodes.append({"id": b, "label": LABELS[b]})
    edges = [{"id": f"{a}->{b}", "source": a, "target": b} for a, b in EDGES[engine]]
    if engine == "graph":
        edges.append({"id": "rerank->query_expansion", "source": "rerank", "target": "query_expansion", "cyclic": True})
    return {"engine": engine, "nodes": nodes, "edges": edges}


class Engine:
    def __init__(self, store: IndexStore, name: str, *, disabled_nodes=(), disabled_edges=(),
                 selected_nodes: set[str] | None = None, blocked_edges: set[str] | None = None):
        self.s, self.name, self.hp = store, name, store.hp
        self.dn, self.de = set(disabled_nodes), set(disabled_edges)
        self.sel, self.blocked = selected_nodes or None, blocked_edges or None
        self.graph = self._build()

    # --- helpers ------------------------------------------------------------
    def _allowed_chunks(self) -> set[str] | None:
        if not self.sel:
            return None
        ids: set[str] = set()
        for n in self.sel:
            ids |= self.s.kg.node_chunks.get(n, set())
        return ids

    def _filter(self, ids: list[str]) -> list[str]:
        ok = self._allowed_chunks()
        return [i for i in ids if ok is None or i in ok]


    def _timed(self, name: str, fn: Callable[[RAGState], dict]):
        def step(state: RAGState) -> dict:
            t0 = time.perf_counter()
            out = {} if name in self.dn else fn(state)
            ms = (time.perf_counter() - t0) * 1000
            g_ms = out.pop("_graph_ms", 0.0)          # graph-traversal time carved out of the node total
            tm = dict(state.get("timings", {}))
            tm[CATEGORY[name]] = tm.get(CATEGORY[name], 0.0) + ms - g_ms
            tm["graph"] = tm.get("graph", 0.0) + g_ms
            tr = list(state.get("trace", [])) + [f"{name}{' (skipped)' if name in self.dn else ''}"]
            return {**out, "timings": tm, "trace": tr}
        return step

    # --- nodes ----------------------------------------------------------------
    def n_retrieve(self, st: RAGState) -> dict:
        qs = st.get("queries") or [st["query"]]
        k = self.hp.top_k * 3 if self.name != "standard" else self.hp.top_k
        lists = [self._filter([c for c, _ in self.s.vector_search(q, k)]) for q in qs]
        if self.name == "standard":
            return {"candidates": {"vector": lists[0]}, "context_ids": lists[0][: self.hp.top_k]}
        fused = rrf(lists, self.hp.rrf_alpha)   # HyDE: merge the N passage result lists
        return {"candidates": {f"hyde_{i}": l for i, l in enumerate(lists)}, "fused": fused,
                "context_ids": [c for c, _ in fused[: self.hp.top_k]]}

    def n_hyde(self, st: RAGState) -> dict:
        n = self.hp.hyde_passages
        if config.OFFLINE:
            expanded = st["query"]
            for k, v in LEXICON.items():
                if k in st["query"].lower():
                    expanded += f" {v}"
            passages = [f"Regarding {HYDE_ANGLES[i % 3]}: {expanded}" for i in range(n)]
        else:
            passages = [llm.complete(
                "Write a short hypothetical passage (<=80 words) in the style of an Indian regulatory "
                f"circular or contract clause that would answer the question, focusing on "
                f"{HYDE_ANGLES[i % len(HYDE_ANGLES)]}. It may be imperfect; it is only used for retrieval.",
                st["query"]) for i in range(n)]
        return {"queries": passages}

    def n_expand(self, st: RAGState) -> dict:
        q = st["query"]
        extra = self.s.kg.match_entities(q)
        if config.OFFLINE:
            alts = [v for k, v in LEXICON.items() if k in q.lower()]
        else:
            alts = llm.complete_json(
                "Rewrite the question into 3 alternative search queries using precise regulatory vocabulary "
                "(statute names, formal terms, entity names). JSON list of strings.", q)
        passes = st.get("loops", 0)
        if passes > 0:      # 2nd pass of the cycle: widen with 1-hop neighbours of the linked entities
            extra = extra + [n for n in self.s.kg.k_hop(extra, 1) if "::" not in n][:6]
        return {"queries": [q] + [f"{q} {a}" for a in alts][:3] + ([" ".join(extra)] if extra else []),
                "loops": passes + 1}

    def n_hybrid(self, st: RAGState) -> dict:
        qs = st.get("queries") or [st["query"]]
        k = self.hp.top_k * 3
        vec = [self._filter([c for c, _ in self.s.vector_search(q, k)]) for q in qs]
        bm = [self._filter([c for c, _ in self.s.bm25_search(q, k)]) for q in qs]
        t0 = time.perf_counter()
        seeds = list({e for q in qs for e in self.s.kg.match_entities(q)})
        hops = self.s.kg.k_hop(seeds, self.hp.hop_radius, self.sel, self.blocked)
        scored = self.s.kg.chunks_for(hops)
        graph_ids = self._filter([c for c, _ in sorted(scored.items(), key=lambda x: -x[1])][:k])
        g_ms = (time.perf_counter() - t0) * 1000
        cands = {"vector": [c for l in vec for c in l], "bm25": [c for l in bm for c in l], "graph": graph_ids}
        return {"candidates": cands, "_graph_ms": g_ms}

    def n_fuse(self, st: RAGState) -> dict:
        lists = [list(dict.fromkeys(v)) for v in st["candidates"].values()]
        return {"fused": rrf(lists, self.hp.rrf_alpha)}

    def n_rerank(self, st: RAGState) -> dict:
        fused = st.get("fused") or [(c, 0.0) for c in dict.fromkeys(
            x for v in st.get("candidates", {}).values() for x in v)]   # fusion toggled off: raw candidates
        texts = {c: self.s.chunks[c].page_content for c, _ in fused}
        return {"fused": fused, "context_ids": rerank(st["query"], fused, texts, self.hp.top_k)}

    def n_synth(self, st: RAGState) -> dict:
        ids = st.get("context_ids") or [c for c, _ in st.get("fused", [])][: self.hp.top_k]
        chunks = [self.s.chunks[i] for i in ids]
        return {"answer": synthesize(st["query"], chunks, st["practice_area"]), "context_ids": ids}

    # --- graph construction ---------------------------------------------------------
    @staticmethod
    def _thin_check(st: RAGState) -> str:
        # Cycle: thin reranked context -> back to query expansion once (`loops` counts expansion passes).
        return "loop" if len(st.get("context_ids", [])) < 2 and st.get("loops", 0) < 2 else "go"

    def _build(self):
        fns = {"retrieve": self.n_retrieve, "hyde_generate": self.n_hyde, "query_expansion": self.n_expand,
               "hybrid_search": self.n_hybrid, "rrf_fusion": self.n_fuse, "rerank": self.n_rerank,
               "synthesis": self.n_synth, "synthesize": self.n_synth}
        g = StateGraph(RAGState)
        edges = EDGES[self.name]
        names = [ENTRY[self.name]] + [b for _, b in edges]
        for n in names:
            g.add_node(n, self._timed(n, fns[n]))
        g.add_edge(START, ENTRY[self.name])
        nxt = {a: b for a, b in edges}
        for n in names:
            b = nxt.get(n)
            edge_on = b is not None and f"{n}->{b}" not in self.de
            if n == "rerank" and self.name == "graph" and edge_on and "rerank->query_expansion" not in self.de:
                g.add_conditional_edges("rerank", self._thin_check, {"loop": "query_expansion", "go": b})
            elif edge_on:
                g.add_edge(n, b)
            else:
                g.add_edge(n, END)
        return g.compile()

    def run(self, query: str, practice_area: str = "tax") -> dict:
        t0 = time.perf_counter()
        out = self.graph.invoke({"query": query, "practice_area": practice_area, "loops": 0,
                                 "timings": {}, "trace": []})
        tm = dict(out.get("timings", {}))
        tm["total"] = (time.perf_counter() - t0) * 1000
        ids = out.get("context_ids", [])
        return {"engine": self.name, "answer": out.get("answer", ""), "context_ids": ids,
                "citations": [{"chunk_id": i, "source": self.s.chunks[i].metadata["source"],
                               "section": self.s.chunks[i].metadata.get("section", "n/a"),
                               "text": self.s.chunks[i].page_content} for i in ids],
                "timings_ms": {k: round(v, 1) for k, v in tm.items()}, "trace": out.get("trace", []),
                "mode": "offline-heuristic" if config.OFFLINE else f"llm:{config.LLM_MODEL}"}
