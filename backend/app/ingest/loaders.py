"""Step 1: load PDF / DOCX / TXT / JSON / CSV into Documents with strict provenance."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from langchain_core.documents import Document

PRACTICE_HINTS = {
    "tax": ("cbdt", "cbic", "gst", "income tax", "transfer pricing"),
    "grc": ("rbi", "kyc", "pmla", "sebi", "brsr", "iso 31000", "audit control"),
    "deals": ("share purchase", "spa", "indemnity", "fema", "companies act", "valuation"),
}


def infer_practice_area(text: str) -> str:
    low = text.lower()
    scores = {a: sum(low.count(h) for h in hs) for a, hs in PRACTICE_HINTS.items()}
    return max(scores, key=scores.get)


def load_file(path: Path) -> list[Document]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:  # spec: UnstructuredPDFLoader; fall back to pypdf when `unstructured` isn't installed
            from langchain_community.document_loaders import UnstructuredPDFLoader

            docs = UnstructuredPDFLoader(str(path)).load()
        except ImportError:
            from langchain_community.document_loaders import PyPDFLoader

            docs = PyPDFLoader(str(path)).load()
    elif ext == ".docx":
        import docx

        d = docx.Document(str(path))
        docs = [Document(page_content="\n".join(p.text for p in d.paragraphs if p.text.strip()))]
    elif ext == ".json":
        docs = [Document(page_content=json.dumps(json.loads(path.read_text()), indent=1))]
    elif ext == ".csv":
        rows = list(csv.DictReader(path.open()))
        docs = [Document(page_content="\n".join("; ".join(f"{k}: {v}" for k, v in r.items()) for r in rows))]
    elif ext in {".txt", ".md"}:
        docs = [Document(page_content=path.read_text())]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    for i, d in enumerate(docs):
        d.metadata.update(
            source=path.name,
            doc_type=ext.lstrip("."),
            page=d.metadata.get("page", i),
            practice_area=infer_practice_area(d.page_content),
        )
    return docs
