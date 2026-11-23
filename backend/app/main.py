"""FastAPI backend for the five UI tabs."""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config
from .eval.runner import run_benchmark
from .index.store import IndexStore
from .rag.pipelines import Engine, workflow_description

UPLOAD_DIR = config.DATA_DIR / "uploads"
ALLOWED = {".pdf", ".docx", ".txt", ".md", ".json", ".csv"}
MAX_BYTES = 25 * 1024 * 1024

app = FastAPI(title="Comparative RAG Benchmark")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

STATE: dict = {"store": None, "selection": {"nodes": [], "edges": []}, "last": {}, "bench": {"status": "idle"},
               "reindex_required": False}
_lock = threading.Lock()


def store() -> IndexStore:
    with _lock:
        if STATE["store"] is None:
            STATE["store"] = IndexStore()
        return STATE["store"]


class ChatReq(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    engine: str = "graph"
    practice_area: str = "tax"
    disabled_nodes: list[str] = []
    disabled_edges: list[str] = []
    use_graph_selection: bool = True


class FilterReq(BaseModel):
    nodes: list[str] = []
    edges: list[str] = []


class HPReq(BaseModel):
    chunk_size: int | None = Field(None, ge=50, le=2000)
    chunk_overlap: int | None = Field(None, ge=0, le=500)
    hyde_passages: int | None = Field(None, ge=1, le=8)
    hop_radius: int | None = Field(None, ge=1, le=4)
    rrf_alpha: int | None = Field(None, ge=1, le=500)
    top_k: int | None = Field(None, ge=1, le=10)


@app.get("/health")
def health():
    return {"ok": True, "mode": "offline-heuristic" if config.OFFLINE else f"llm:{config.LLM_MODEL}",
            "embed_model": config.EMBED_MODEL, "chunks": len(store().chunks)}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(f.filename or "upload").name)   # strip any path components
        if Path(name).suffix.lower() not in ALLOWED:
            raise HTTPException(400, f"Unsupported file type: {name}")
        data = await f.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{name} exceeds {MAX_BYTES // 2**20} MB")
        dest = UPLOAD_DIR / name
        dest.write_bytes(data)
        saved.append(dest)
    n = store().ingest_paths(saved)
    return {"files": [p.name for p in saved], "new_chunks": n, "total_chunks": len(store().chunks)}


@app.post("/ingest/sample")
def ingest_sample():
    n = store().ingest_paths(sorted(config.SAMPLE_DIR.glob("*.txt")))
    return {"new_chunks": n, "total_chunks": len(store().chunks)}


@app.get("/graph")
def graph():
    return {**store().kg.to_json(), "selection": STATE["selection"]}


@app.post("/graph/filter")
def graph_filter(req: FilterReq):
    STATE["selection"] = {"nodes": req.nodes, "edges": req.edges}
    return STATE["selection"]


@app.post("/chat")
def chat(req: ChatReq):
    if req.engine not in config.ENGINES:
        raise HTTPException(400, "engine must be one of " + ", ".join(config.ENGINES))
    if req.practice_area not in config.PRACTICE_AREAS:
        raise HTTPException(400, "unknown practice_area")
    s = store()
    if not s.chunks:
        raise HTTPException(409, "Index is empty - upload documents or POST /ingest/sample first")
    sel = STATE["selection"] if req.use_graph_selection else {"nodes": [], "edges": []}
    # Edges the user unselected in the graph canvas are blocked in traversal only when some edges are selected.
    all_edges = {e["id"] for e in s.kg.to_json()["edges"]}
    blocked = (all_edges - set(sel["edges"])) if sel["edges"] else None
    eng = Engine(s, req.engine, disabled_nodes=req.disabled_nodes, disabled_edges=req.disabled_edges,
                 selected_nodes=set(sel["nodes"]) or None, blocked_edges=blocked)
    try:
        res = eng.run(req.query, req.practice_area)
    except Exception as e:  # surface LLM/key errors to the UI without a stack dump
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    STATE["last"][req.engine] = {"timings_ms": res["timings_ms"], "query": req.query, "ts": time.time()}
    return res


@app.get("/workflow/{engine}")
def workflow(engine: str):
    if engine not in config.ENGINES:
        raise HTTPException(404, "unknown engine")
    return workflow_description(engine)


@app.get("/hyperparameters")
def get_hp():
    return {**store().hp.as_dict(), "reindex_required": STATE["reindex_required"]}


@app.put("/hyperparameters")
def put_hp(req: HPReq):
    hp = store().hp
    for k, v in req.model_dump(exclude_none=True).items():
        if k in ("chunk_size", "chunk_overlap") and getattr(hp, k) != v:
            STATE["reindex_required"] = True    # chunking only applies to newly ingested documents
        setattr(hp, k, v)
    return get_hp()


@app.get("/telemetry")
def telemetry():
    s = store()
    from .eval.metrics import GOLD_ER_LINKS, er_resolution_accuracy

    bench = STATE["bench"].get("result")
    return {"hyperparameters": get_hp(), "last_queries": STATE["last"],
            "er_accuracy": er_resolution_accuracy(s.kg, GOLD_ER_LINKS) if s.chunks else None,
            "benchmark_summary": ({e: m["all"] for e, m in bench["matrix"].items()} if bench else None),
            "graph": {"nodes": s.kg.g.number_of_nodes(), "edges": s.kg.g.number_of_edges()},
            "mode": "offline-heuristic" if config.OFFLINE else f"llm:{config.LLM_MODEL}"}


def _bench_job():
    try:
        STATE["bench"] = {"status": "done", "result": run_benchmark(store())}
        out = config.DATA_DIR / "benchmark_results.json"
        out.write_text(json.dumps(STATE["bench"]["result"], indent=1))
    except Exception as e:
        STATE["bench"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}


@app.post("/benchmark/run")
def bench_run():
    if STATE["bench"].get("status") == "running":
        raise HTTPException(409, "benchmark already running")
    if not store().chunks:
        raise HTTPException(409, "Index is empty - ingest first")
    STATE["bench"] = {"status": "running"}
    threading.Thread(target=_bench_job, daemon=True).start()
    return STATE["bench"]


@app.get("/benchmark/results")
def bench_results():
    return {**STATE["bench"], "reference": REFERENCE}


# Figures from the architecture document - EXPECTED values, shown next to (never instead of) measurements.
REFERENCE = {
    "latency_ms": {"standard": 450, "hyde": 1200, "graph": 850},
    "hallucination_risk": {"standard": "Moderate (out-of-context chunks)", "hyde": "Moderate (synthetic passage drift)",
                           "graph": "Lowest (strict entity-relation grounding)"},
    "best_use_case": {"standard": "Single document lookup (SOPs)", "hyde": "Vague regulatory inquiries",
                      "graph": "Complex Tax/M&A due diligence"},
}
