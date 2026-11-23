"""Run all three engines over the gold QA set and aggregate by query type."""
from __future__ import annotations

import json
import statistics as st

from .. import config
from ..rag.pipelines import Engine
from .metrics import GOLD_ER_LINKS, er_resolution_accuracy, faithfulness, retrieval_context_recall

TYPES = ("simple", "multi_hop", "vocab_mismatch")


def run_benchmark(store, qa_path=None) -> dict:
    qa = json.loads((qa_path or config.EVAL_DIR / "gold_qa.json").read_text())
    rows = []
    for engine in config.ENGINES:
        eng = Engine(store, engine)
        for item in qa:
            res = eng.run(item["q"], item["practice_area"])
            ctx = [c["text"] for c in res["citations"]]
            rows.append({"engine": engine, "id": item["id"], "type": item["type"],
                         "rcr": retrieval_context_recall(item["facts"], ctx),
                         "faithfulness": faithfulness(res["answer"], ctx),
                         "latency_ms": res["timings_ms"]["total"], "timings_ms": res["timings_ms"],
                         "answer": res["answer"]})
    matrix: dict = {}
    for engine in config.ENGINES:
        matrix[engine] = {}
        for t in (*TYPES, "all"):
            sel = [r for r in rows if r["engine"] == engine and (t == "all" or r["type"] == t)]
            matrix[engine][t] = {
                "rcr": round(st.mean(r["rcr"] for r in sel), 3),
                "faithfulness": round(st.mean(r["faithfulness"] for r in sel), 3),
                "latency_ms": round(st.mean(r["latency_ms"] for r in sel), 1),
                "hallucination_risk": round(1 - st.mean(r["faithfulness"] for r in sel), 3),
                "n": len(sel)}
    return {"mode": "offline-heuristic" if config.OFFLINE else f"llm:{config.LLM_MODEL}",
            "hyperparameters": store.hp.as_dict(),
            "er_accuracy": er_resolution_accuracy(store.kg, GOLD_ER_LINKS),
            "matrix": matrix, "rows": rows}
