"""Steps 2-3: clause-aware chunking measured in bge sub-word tokens (600 / 60 by default)."""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import config

# Legal clause boundaries are tried before generic paragraph/sentence separators.
CLAUSE_SEPARATORS = [r"\n(?=Section \d+\.)", r"\n(?=Rule \d+\.)", r"\n(?=Clause \d+\.)", r"\n(?=Para \d+\.)",
                     "\n\n", "\n", ". ", " ", ""]
SECTION_RE = re.compile(r"^(Section|Rule|Clause|Para)\s+(\d+)\.", re.M)


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(config.EMBED_MODEL)
    except Exception:  # offline / hash-embedder runs: approximate with whitespace tokens
        return None


def count_tokens(text: str) -> int:
    tok = _tokenizer()
    return len(tok.encode(text, add_special_tokens=False)) if tok else len(text.split())


def split_documents(docs: list[Document], hp: config.Hyperparameters | None = None) -> list[Document]:
    hp = hp or config.Hyperparameters()
    splitter = RecursiveCharacterTextSplitter(
        separators=CLAUSE_SEPARATORS, is_separator_regex=True, keep_separator=True,
        chunk_size=hp.chunk_size, chunk_overlap=hp.chunk_overlap, length_function=count_tokens,
    )
    out: list[Document] = []
    for doc in docs:
        for n, piece in enumerate(splitter.split_documents([doc])):
            m = SECTION_RE.search(piece.page_content)
            piece.metadata["chunk_id"] = f"{doc.metadata['source']}::{doc.metadata.get('page', 0)}::{n}"
            piece.metadata["section"] = f"{m.group(1)} {m.group(2)}" if m else "n/a"
            piece.metadata["sections"] = ", ".join(f"{a} {b}" for a, b in SECTION_RE.findall(piece.page_content)) or "n/a"
            piece.metadata["tokens"] = count_tokens(piece.page_content)
            out.append(piece)
    return out
