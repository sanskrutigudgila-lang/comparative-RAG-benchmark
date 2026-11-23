"""Step 5(b)/6: Entity -> Relation -> Target adjacency index on NetworkX, with k-hop traversal
and Node2Vec structural embeddings. (A Neo4j adapter would implement the same 4 methods.)"""
from __future__ import annotations

import networkx as nx
from langchain_core.documents import Document

from ..ingest.entities import canonical, extract_triples


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.node_chunks: dict[str, set[str]] = {}   # entity -> chunk_ids that mention it
        self.embeddings: dict[str, list[float]] = {}

    def add_chunk(self, chunk: Document):
        cid = chunk.metadata["chunk_id"]
        for h, r, t in extract_triples(chunk):
            for n in (h, t):
                self.g.add_node(n)
                self.node_chunks.setdefault(n, set()).add(cid)
            if not self.g.has_edge(h, t, key=r):
                self.g.add_edge(h, t, key=r, relation=r)

    def build(self, chunks: list[Document]):
        for c in chunks:
            self.add_chunk(c)

    # --- retrieval helpers -------------------------------------------------
    def match_entities(self, query: str) -> list[str]:
        """Link query text to graph nodes using the same alias normalisation as ingestion."""
        q = query.lower()
        hits = [n for n in self.g.nodes if len(n) > 2 and "::" not in n and n.lower() in q]
        from ..ingest.entities import ALIASES

        for alias, canon in ALIASES.items():
            if f" {alias} " in f" {q.replace('?', ' ').replace(',', ' ')} " and canon in self.g:
                hits.append(canon)
        return list(dict.fromkeys(hits))

    def k_hop(self, seeds: list[str], k: int, allowed_nodes: set[str] | None = None,
              blocked_edges: set[str] | None = None) -> dict[str, int]:
        """Return {node: hop_distance} within k hops (undirected), optionally limited to a UI selection."""
        if blocked_edges:
            und = nx.Graph()
            und.add_nodes_from(self.g.nodes)
            und.add_edges_from((u, v) for u, v, r in self.g.edges(keys=True) if f"{u}|{r}|{v}" not in blocked_edges)
        else:
            und = self.g.to_undirected(as_view=True)
        dist: dict[str, int] = {}
        for s in seeds:
            if s not in und:
                continue
            for n, d in nx.single_source_shortest_path_length(und, s, cutoff=k).items():
                if allowed_nodes is not None and n not in allowed_nodes:
                    continue
                dist[n] = min(d, dist.get(n, d))
        return dist

    def chunks_for(self, nodes: dict[str, int]) -> dict[str, float]:
        """Score chunk ids by sum of 1/(1+hop) over the nodes that mention them."""
        scores: dict[str, float] = {}
        for n, d in nodes.items():
            for cid in self.node_chunks.get(n, ()):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (1 + d)
        return scores

    # --- Node2Vec-style structural embeddings (step 4) -----------------------------
    # The `node2vec` pip package needs gensim, which has no Python 3.14 wheel. Same method, in numpy:
    # second-order biased random walks (return p / in-out q), then SVD of the PPMI co-occurrence
    # matrix, which is the closed-form equivalent of skip-gram with negative sampling.
    def train_node2vec(self, dim: int = 32, walk_length: int = 10, num_walks: int = 20,
                       window: int = 4, p: float = 1.0, q: float = 0.5, seed: int = 7):
        import numpy as np

        und = nx.Graph(self.g)
        nodes = list(und.nodes)
        if len(nodes) < 3:
            return
        idx = {n: i for i, n in enumerate(nodes)}
        rng = np.random.default_rng(seed)
        co = np.zeros((len(nodes), len(nodes)))
        for _ in range(num_walks):
            for start in nodes:
                walk = [start]
                while len(walk) < walk_length:
                    cur = walk[-1]
                    nbrs = list(und.neighbors(cur))
                    if not nbrs:
                        break
                    if len(walk) == 1:
                        walk.append(nbrs[rng.integers(len(nbrs))])
                        continue
                    prev = walk[-2]
                    w = np.array([1 / p if n == prev else 1.0 if und.has_edge(n, prev) else 1 / q for n in nbrs])
                    walk.append(nbrs[rng.choice(len(nbrs), p=w / w.sum())])
                for i, a in enumerate(walk):
                    for b in walk[i + 1:i + 1 + window]:
                        co[idx[a], idx[b]] += 1
                        co[idx[b], idx[a]] += 1
        total = co.sum()
        if total == 0:
            return
        row, col = co.sum(1, keepdims=True), co.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ppmi = np.maximum(np.log(co * total / (row * col)), 0)
        ppmi[~np.isfinite(ppmi)] = 0
        u, s_, _ = np.linalg.svd(ppmi, full_matrices=False)
        d = min(dim, len(nodes))
        emb = u[:, :d] * np.sqrt(s_[:d])
        self.embeddings = {n: emb[idx[n]].tolist() for n in nodes}

    # --- serialisation for the UI ----------------------------------------------
    def to_json(self) -> dict:
        deg = dict(self.g.degree())
        return {
            "nodes": [{"id": n, "kind": "chunk_ref" if "::" in n else "entity", "degree": deg[n],
                       "chunks": len(self.node_chunks.get(n, ()))} for n in self.g.nodes],
            "edges": [{"id": f"{u}|{r}|{v}", "source": u, "target": v, "relation": r}
                      for u, v, r in self.g.edges(keys=True)],
        }
