import os
os.environ["RAG_EMBED_MODEL"] = "hash"
os.environ["RAG_OFFLINE"] = "1"
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from app import config
from app.index.store import IndexStore


@pytest.fixture(scope="session")
def store():
    s = IndexStore(collection="test")
    s.ingest_paths(sorted(config.SAMPLE_DIR.glob("*.txt")))
    return s
