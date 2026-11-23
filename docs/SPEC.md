# PwC India Advisory Practice — Comparative Benchmark: Standard RAG vs HyDE RAG vs Graph RAG

LangChain + LangGraph framework specification for domain-grounded QA across Tax, Risk Advisory and Deal Services.
(Source: "Confidential Architecture Document", 3 pages. Kept verbatim in substance; this is the reference for the build.)

## Executive summary
Enterprise compliance, audit and deal advisories deal with dense, interconnected, multi-layered regulatory frameworks
(CBDT/CBIC tax notifications, RBI KYC/PMLA rules, corporate legal contracts). Standard vector RAG struggles with
multi-hop relational queries and domain vocabulary mismatch; HyDE bridges semantic gaps but lacks entity awareness.

## 1. System prompt (CO-STAR) and domain guardrails
- **C** Context: Senior Principal Advisory AI Assistant serving PwC India domain consultants.
- **O** Objective: accurate, verifiable, cited compliance analysis from indexed enterprise corpora.
- **S** Style: professional, analytical, structured, auditable corporate advisory.
- **T** Tone: authoritative, objective, risk-aware, precise.
- **A** Audience: Partners, Enterprise Clients, Tax Controllers, Chief Risk Officers, Audit Committees.
- **R** Response: structured markdown with inline statutory/clause citations and explicit gap disclosures.

| Practice area | Core regulatory focus | Enforced guardrails |
|---|---|---|
| 1. Tax & Regulatory Services | Direct tax (CBDT notifications, Income Tax Act); indirect tax (CBIC circulars, 57th GST Council rulings, ISD rules, e-invoicing) | Cite exact Section/Rule numbers. Distinguish prospective amendments from retrospective clarifications. |
| 2. Risk Advisory & Governance (GRC) | RBI Master Directions (KYC/PMLA), SEBI BRSR, ISO 31000, internal audit controls | Flag non-compliance penalties immediately. Strictly map controls to specific regulatory obligations. |
| 3. Deal Advisory & Corporate Finance | M&A SPAs, Companies Act 2013 valuation standards, FEMA/FCRA foreign investment rules | Highlight indemnities, MAC clauses, financial exposure thresholds. |

## 2. Graph RAG 6-step architecture (LangGraph)
1. **Data format & ingestion** — PDF circulars, docx SOPs, JSON/tabular tax tables; UnstructuredPDFLoader; strict provenance metadata.
2. **Semantic & entity chunking** — RecursiveCharacterTextSplitter (600 tokens / 60 overlap) + LLM entity-relation extraction; never split a legal clause.
3. **Sub-word tokenization** — HF tokenizer matching BAAI/bge-large-en-v1.5 limits; normalised entity nodes (CBDT = Income Tax Dept = Tax Authority).
4. **Multi-modal embedding** — 1024-dim bge-large-en-v1.5 for chunks; Node2Vec for entity-relation graph nodes.
5. **Dual indexing** — (a) vector HNSW index; (b) graph adjacency index (Entity -> Relation -> Target) for multi-hop traversal.
6. **Hybrid vector DB & graph engine** — Qdrant/ChromaDB for vectors, Neo4j/NetworkX for the KG; LangGraph orchestrates cyclic sub-graph traversal + vector retrieval.

## 3. UI: five tabs
1. Data ingestion & multi-RAG chat console (drag-drop PDF/DOCX, practice-area dropdown, engine selector, chat with inline citations).
2. Interactive spherical 3D knowledge graph canvas (click to select/unselect nodes and edges to filter retrieval).
3. LangGraph workflow canvas (Query Expansion -> Hybrid Vector/Graph Search -> RRF Fusion -> Rerank -> Synthesis; toggle nodes/edges).
4. Hyperparameters & KPI telemetry dashboard.
5. Comparative analysis & live performance benchmark (side-by-side matrix, all three engines).

## 4. Hyperparameters and KPIs
Hyperparameters: (1) chunk size/overlap 600/60; (2) Multi-HyDE passage count N=3; (3) graph hop radius k=2; (4) RRF constant alpha=60 (BM25 + vector + graph).
KPIs: (1) Retrieval Context Recall; (2) Faithfulness/hallucination score 0.0-1.0; (3) Entity-relation resolution accuracy; (4) end-to-end latency ms (retrieval, graph traversal, LLM).

## 5. Reference comparison (expected, from the document — not measured)
| Metric | Standard RAG | Multi-hypothesis HyDE | Graph RAG |
|---|---|---|---|
| Simple keyword queries | High precision / low latency | Moderate (HyDE overhead) | Moderate (graph overhead) |
| Multi-hop relational | Poor (misses indirect relationships) | Moderate (semantic expansion) | Superior (k-hop edges) |
| Domain vocabulary mismatch | Poor (exact vector distance) | Superior (synthesises domain terms) | High (entity resolution maps synonyms) |
| Avg latency | ~450 ms | ~1,200 ms | ~850 ms |
| Hallucination risk | Moderate (out-of-context chunks) | Moderate (synthetic passage drift) | Lowest (strict entity-relation grounding) |
| Best use case | Single-document lookup (SOPs) | Vague regulatory inquiries | Complex tax / M&A due diligence |
