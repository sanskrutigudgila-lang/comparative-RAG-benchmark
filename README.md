# Comparative RAG Benchmark — Standard vs HyDE vs Graph RAG

Implementation of the PwC India Advisory architecture document (`docs/SPEC.md`): three LangChain + LangGraph
pipelines, a CO-STAR prompt with practice-area guardrails, and a 5-tab UI.

## Run
```bash
python3 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.15.0+cpu"   # CPU torch; skip if you have a GPU
.venv/bin/pip install -r backend/requirements.txt

export ANTHROPIC_API_KEY=...            # real LLM mode (claude-sonnet-5-5). Never commit it.
cd backend && ../.venv/bin/uvicorn app.main:app --port 8000

cd frontend && npm install && npm run dev       # http://localhost:5173
# if port 8000 is taken:  uvicorn ... --port 9000  and  VITE_API_TARGET=http://127.0.0.1:9000 npm run dev
```
Then: Tab 1 -> "Load sample corpus", ask questions; Tab 5 -> "Run live benchmark".

Modes
- **LLM mode** (`ANTHROPIC_API_KEY` set): Claude does entity extraction, query expansion, HyDE passages, synthesis and the faithfulness judge. Calls are cached by prompt hash in `backend/data/llm_cache`.
- **Offline heuristic mode** (no key, or `RAG_OFFLINE=1`): deterministic extraction/expansion and extractive answers so everything runs without a key. Every result is labelled with its mode. Faithfulness is not meaningful here.
- `RAG_EMBED_MODEL=hash` swaps bge-large-en-v1.5 (about 1.3 GB download) for a tiny hashing embedder - used by the tests.

## Layout
`backend/app/ingest` loaders, clause-aware 600/60 token chunking, entity-relation extraction + alias normalisation ·
`backend/app/index` Chroma (HNSW) + BM25 + NetworkX graph (k-hop) + Node2Vec-style embeddings ·
`backend/app/rag` RRF (alpha=60), rerank, synthesis, the three LangGraph pipelines (toggleable nodes/edges; Graph RAG has a real cycle) ·
`backend/app/eval` RCR, faithfulness, ER accuracy, latency + benchmark runner · `frontend/src/tabs` the five tabs.

## Tests
`cd backend && RAG_EMBED_MODEL=hash RAG_OFFLINE=1 ../.venv/bin/python -m pytest -q tests`  (12 tests)

## Known limitations
- The sample corpus is **synthetic** (labelled as such) and small (9 gold questions); numbers are indicative, not statistically meaningful.
- The spec's ~450 / ~850 / ~1,200 ms figures are shown as *expected targets* next to measurements, never as results.
- `node2vec` (needs gensim, no Python 3.14 wheel) is replaced by an equivalent numpy implementation (biased walks + PPMI-SVD).
- Neo4j and Qdrant are not wired in; NetworkX and Chroma are the default stores behind small interfaces.
- The reranker is lexical + RRF blend, not a cross-encoder.
- Not yet run: the full benchmark in LLM mode with bge-large embeddings (no API key / model download in the build environment).
