from fastapi.testclient import TestClient
from app import main


def test_api_flow(store):
    main.STATE["store"] = store
    c = TestClient(main.app)
    assert c.get("/health").json()["ok"]
    g = c.get("/graph").json()
    assert g["nodes"] and g["edges"]
    r = c.post("/chat", json={"query": "What is the indemnity cap?", "engine": "graph", "practice_area": "deals"})
    assert r.status_code == 200 and r.json()["citations"]
    assert c.post("/chat", json={"query": "x", "engine": "nope"}).status_code == 400
    assert c.get("/workflow/graph").json()["edges"]
    assert c.put("/hyperparameters", json={"hop_radius": 3}).json()["hop_radius"] == 3
    c.put("/hyperparameters", json={"hop_radius": 2})
    assert c.put("/hyperparameters", json={"top_k": 99}).status_code == 422


def test_upload_rejects_bad_type_and_traversal(store):
    main.STATE["store"] = store
    c = TestClient(main.app)
    assert c.post("/ingest", files={"files": ("evil.exe", b"x")}).status_code == 400
    r = c.post("/ingest", files={"files": ("../../etc/passwd.txt", b"Clause 1. Test clause about RBI penalty.")})
    assert r.status_code == 200 and r.json()["files"] == ["passwd.txt"]
