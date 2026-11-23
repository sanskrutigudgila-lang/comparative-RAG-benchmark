"""CO-STAR synthesis with practice-area guardrails and inline citations."""
from __future__ import annotations

import re

from langchain_core.documents import Document

from .. import config, llm
from ..prompts import system_prompt
from ..index.store import tokenize


def fmt_context(chunks: list[Document]) -> str:
    return "\n\n".join(
        f"[source: {c.metadata['source']} | clauses in this passage: {c.metadata.get('sections', 'n/a')}]\n"
        f"(cite the exact clause label that precedes the text you rely on)\n{c.page_content}" for c in chunks
    )


def synthesize(query: str, chunks: list[Document], practice_area: str) -> str:
    if not chunks:
        return ("**Insufficient context.**\n\n### Gaps & Disclosures\n- No indexed material was retrieved "
                "for this query; no claim can be made.")
    if config.OFFLINE:
        return _extractive(query, chunks)
    return llm.complete(system_prompt(practice_area),
                        f"CONTEXT:\n{fmt_context(chunks)}\n\nQUESTION: {query}")


def _extractive(query: str, chunks: list[Document]) -> str:
    """Offline fallback: pick the best-matching sentences verbatim (zero generated claims)."""
    qt = set(tokenize(query))
    sents = []
    clause = re.compile(r"(Section|Rule|Clause|Para)\s+(\d+)\.")
    for c in chunks:
        label = c.metadata.get("section", "n/a")
        for line in c.page_content.splitlines():          # clauses are one per line in the sample corpus
            m = clause.match(line.strip())
            if m:
                label = f"{m.group(1)} {m.group(2)}"      # exact clause this sentence belongs to
            for s in re.split(r"(?<=[.;])\s+", line):
                if len(s) > 25 and not s.startswith("[SYNTHETIC"):
                    sents.append((len(qt & set(tokenize(s))), re.sub(r"^(Section|Rule|Clause|Para)\s+\d+\.\s*", "", s).strip(), c, label))   # strip only the leading label
    sents.sort(key=lambda x: -x[0])
    lines = [f"- {s} [source: {c.metadata['source']} | {lab}]" for _, s, c, lab in sents[:5]]
    return ("*(offline extractive mode - no LLM)*\n\n### Analysis\n" + "\n".join(lines) +
            "\n\n### Gaps & Disclosures\n- Answer assembled from retrieved text only; verify against source documents.")
