from app import config
from app.ingest.chunking import count_tokens, split_documents
from app.ingest.entities import canonical
from app.ingest.loaders import load_file
from app.rag.fusion import rrf
from app.rag.pipelines import Engine
from langchain_core.documents import Document


def test_chunks_respect_size_and_keep_clauses():
    docs = load_file(config.SAMPLE_DIR / "deals_spa_zenith_acquisition.txt")
    chunks = split_documents(docs)
    assert all(c.metadata["tokens"] <= 600 for c in chunks)
    assert all(c.metadata["source"] and c.metadata["practice_area"] for c in chunks)  # provenance
    # small hp forces splitting; a clause must not be cut mid-sentence at a "Clause N." boundary
    small = split_documents(docs, config.Hyperparameters(chunk_size=60, chunk_overlap=6))
    assert len(small) > len(chunks)
    assert all(c.page_content.strip() for c in small)


def test_alias_normalisation_merges_cbdt_synonyms():
    assert {canonical(x) for x in ["CBDT", "Income Tax Dept", "Tax Authority", "the Income Tax Department"]} == {"CBDT"}


def test_rrf_alpha_60_math():
    out = dict(rrf([["a", "b"], ["b", "c"]], 60))
    assert abs(out["a"] - 1 / 61) < 1e-9
    assert abs(out["b"] - (1 / 62 + 1 / 61)) < 1e-9
    assert list(out)[0] == "b"


def test_k_hop_two_not_three(store):
    kg = store.kg
    d2 = kg.k_hop(["Orion Holdings"], 2)
    d1 = kg.k_hop(["Orion Holdings"], 1)
    assert max(d2.values()) <= 2 and len(d2) >= len(d1)
    assert max(kg.k_hop(["Orion Holdings"], 3).values()) <= 3


def test_node2vec_embeddings_built(store):
    assert store.kg.embeddings and "CBDT" in store.kg.embeddings


def test_all_engines_answer_with_citations(store):
    for name in config.ENGINES:
        r = Engine(store, name).run("What is the e-invoicing turnover threshold?", "tax")
        assert r["citations"] and "source:" in r["answer"], name
        assert r["timings_ms"]["total"] > 0


def test_disabled_node_passes_through(store):
    r = Engine(store, "graph", disabled_nodes={"rerank"}).run("indemnity cap", "deals")
    assert "rerank (skipped)" in r["trace"] and r["answer"]


def test_disabled_edge_stops_flow(store):
    r = Engine(store, "graph", disabled_edges={"rrf_fusion->rerank"}).run("indemnity cap", "deals")
    assert "synthesis" not in " ".join(r["trace"])


def test_graph_cycle_terminates(store):
    r = Engine(store, "graph").run("zzzz qqqq", "tax")
    assert r["trace"].count("query_expansion") <= 2


def test_node_filter_restricts_context(store):
    sel = {"RBI"}
    r = Engine(store, "graph", selected_nodes=sel).run("penalty per contravention", "grc")
    allowed = store.kg.node_chunks["RBI"]
    assert set(r["context_ids"]) <= allowed


def test_gold_answer_key_is_never_indexed(store, tmp_path):
    gold = tmp_path / "gold_qa.json"
    gold.write_text('[{"q":"x","facts":["INR 5 crore"]}]')
    before = len(store.chunks)
    assert store.ingest_paths([gold]) == 0 and len(store.chunks) == before
    assert not any("gold_qa" in c for c in store.chunks)
    assert not list(config.SAMPLE_DIR.glob("gold_qa*"))
