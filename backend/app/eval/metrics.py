"""The 4 KPIs: Retrieval Context Recall, Faithfulness, Entity-Relation resolution accuracy, latency."""
from __future__ import annotations

import re

from .. import config, llm
from ..index.store import tokenize


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower())


def retrieval_context_recall(facts: list[str], context_texts: list[str]) -> float:
    """Share of ground-truth facts (exact normalised substrings) present in the retrieved chunks."""
    blob = _norm(" ".join(context_texts))
    return sum(_norm(f) in blob for f in facts) / len(facts) if facts else 0.0


def faithfulness(answer: str, context_texts: list[str]) -> float:
    """Supported-claims ratio in [0,1]. LLM judge online; lexical-support proxy offline."""
    claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n[-*] ", answer) if len(s.strip()) > 30
              and not s.strip().startswith(("#", "*("))]
    if not claims:
        return 0.0
    if not config.OFFLINE:
        verdicts = llm.complete_json(
            "You are a strict fact-checker. For each numbered claim say whether it is fully supported by the "
            'CONTEXT. Output JSON list of booleans, one per claim, same order.',
            "CONTEXT:\n" + "\n---\n".join(context_texts) + "\n\nCLAIMS:\n" +
            "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims)))
        return sum(bool(v) for v in verdicts) / len(claims)
    ctx = set(tokenize(" ".join(context_texts)))
    sup = 0
    for c in claims:
        toks = [t for t in tokenize(re.sub(r"\[source:[^\]]*\]", "", c)) if len(t) > 2]
        sup += bool(toks) and sum(t in ctx for t in toks) / len(toks) >= 0.8
    return sup / len(claims)


def er_resolution_accuracy(kg, gold_links: list[tuple[str, str]]) -> float:
    """Fraction of gold (alias, canonical) pairs for which the alias resolves to a node present in the KG."""
    from ..ingest.entities import canonical

    if not gold_links:
        return 0.0
    return sum(canonical(a) == c and c in kg.g for a, c in gold_links) / len(gold_links)


# Gold alias -> canonical entity pairs for the synthetic corpus.
GOLD_ER_LINKS = [("Income Tax Dept", "CBDT"), ("Tax Authority", "CBDT"), ("Income Tax Department", "CBDT"),
                 ("Input Service Distributor", "ISD"), ("Suspicious Transaction Report", "STR"),
                 ("Material Adverse Change", "MAC"), ("Reserve Bank", "RBI"), ("Financial Intelligence Unit", "FIU")]
