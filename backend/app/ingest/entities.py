"""Steps 2-3: entity-relation extraction + canonical entity normalisation.

Online mode: Claude extracts (head, relation, tail) triples. Offline mode: a deterministic
pattern/gazetteer extractor so the pipeline stays runnable and testable without a key.
"""
from __future__ import annotations

import re

from langchain_core.documents import Document

from .. import config, llm

# Alias -> canonical node (spec step 3: CBDT = Income Tax Dept = Tax Authority).
ALIASES = {
    "cbdt": "CBDT", "income tax department": "CBDT", "income tax dept": "CBDT", "tax authority": "CBDT",
    "cbic": "CBIC", "central board of indirect taxes": "CBIC",
    "rbi": "RBI", "reserve bank": "RBI", "reserve bank of india": "RBI", "central bank": "RBI",
    "fiu": "FIU", "financial intelligence unit": "FIU",
    "isd": "ISD", "input service distributor": "ISD",
    "str": "STR", "suspicious transaction report": "STR",
    "cdd": "CDD", "customer due diligence": "CDD", "edd": "EDD", "enhanced due diligence": "EDD",
    "mac": "MAC", "material adverse change": "MAC",
    "buyer": "Orion Holdings", "acquirer": "Orion Holdings", "seller": "Zenith Pharma Ltd",
    "parent": "Zenith Pharma Ltd", "head office": "Zenith Pharma Ltd",
    "pmla": "PMLA", "pml rules": "PMLA", "fema": "FEMA",
}
ORG_RE = re.compile(r"\b((?:[A-Z][a-z]+ )+(?:Ltd|Pvt Ltd|Holdings|Finance Ltd|Exports Pvt Ltd))\b")
CODE_RE = re.compile(r"\b(IA-\d+|(?:Section|Rule|Para|Clause) \d+[A-Z]?(?: of the [A-Z][\w ]+?(?:Act|Rules))?)")
ROLE_RE = re.compile(r"\b(Chief Compliance Officer)\b")
GAZETTEER = ["CBDT", "CBIC", "RBI", "FIU", "ISD", "STR", "CDD", "EDD", "MAC", "PMLA", "FEMA", "GST Council",
             "Income Tax Act", "CGST Act", "Banking Regulation Act", "Companies Act 2013"]


def canonical(name: str) -> str:
    n = re.sub(r"\s+", " ", name.strip().strip(".,;()"))
    n = re.sub(r"^(the|The) ", "", n)
    return ALIASES.get(n.lower(), n)


def _offline_triples(chunk: Document) -> list[tuple[str, str, str]]:
    text = chunk.page_content
    sec = chunk.metadata.get("section", "n/a")
    src = chunk.metadata["source"]
    orgs = [canonical(o) for o in ORG_RE.findall(text)]
    codes = [canonical(c) for c in CODE_RE.findall(text)]
    roles = ROLE_RE.findall(text)
    low = text.lower()
    terms = [g for g in GAZETTEER if g.lower() in low] + [
        v for k, v in ALIASES.items() if len(k) > 4 and k in low   # alias hit -> canonical node
    ]
    ents = list(dict.fromkeys(orgs + terms + roles))
    triples = [(f"{src}::{sec}", "mentions", e) for e in ents]
    triples += [(f"{src}::{sec}", "cites", c) for c in dict.fromkeys(codes)]
    # co-occurrence inside one clause -> relation edge (what makes multi-hop traversal possible)
    for i, a in enumerate(ents):
        for b in ents[i + 1:]:
            triples.append((a, "related_to", b))
    for pat, rel in ((r"indemnif\w+", "indemnifies"), (r"subsidiary of", "subsidiary_of"),
                     (r"acquires", "acquires"), (r"accountable", "accountable_for"), (r"maps to", "maps_to")):
        if re.search(pat, low) and len(ents) >= 2:
            triples.append((ents[0], rel, ents[1]))
    return triples


def extract_triples(chunk: Document) -> list[tuple[str, str, str]]:
    if config.OFFLINE:
        raw = _offline_triples(chunk)
    else:
        data = llm.complete_json(
            "You extract entity-relation triples from regulatory/legal text for a knowledge graph. "
            'Output JSON: [{"head":str,"relation":str,"tail":str}]. Use short relation verbs. '
            "Keep statutory references (e.g. 'Rule 39 of the CGST Rules') as single entities. "
            "Do not split legal clauses.",
            chunk.page_content,
        )
        src = f"{chunk.metadata['source']}::{chunk.metadata.get('section', 'n/a')}"
        raw = [(d["head"], d["relation"], d["tail"]) for d in data] + [(src, "mentions", d["head"]) for d in data]
    return list(dict.fromkeys((canonical(h), r.strip().lower().replace(" ", "_"), canonical(t)) for h, r, t in raw))
