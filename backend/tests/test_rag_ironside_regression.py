import json

import pytest

from dev.rag import lightweight_retriever


CASES = [
    ("ISM-CNC-001 bearing wear", "ISM-CNC-001"),
    ("ISM-HYD-002 hydraulic pump cavitation", "ISM-HYD-002"),
    ("ISM-GBX-003 gear pitting", "ISM-GBX-003"),
    ("ISM-CMR-004 air-end bearing fatigue", "ISM-CMR-004"),
]


@pytest.mark.parametrize(("query", "asset_id"), CASES)
def test_asset_queries_return_deterministic_ironside_evidence(client, query, asset_id):
    first = client.post("/api/v1/rag/query", json={"query": query, "top_k": 3})
    second = client.post("/api/v1/rag/query", json={"query": query, "top_k": 3})
    assert first.status_code == 200
    body = first.json()
    assert body["abstained"] is False
    assert 1 <= len(body["sources"]) <= 3
    assert body["corpus_version"] == "2.0.0-ironside"
    assert body["retrieval_mode"] == "resource_safe_lexical"
    assert body["sources"] == second.json()["sources"]
    assert asset_id in (body["answer"] + json.dumps(body["sources"]))
    assert "turbofan unit" not in body["answer"].lower()
    scores = [source["relevance_score"] for source in body["sources"]]
    assert scores == sorted(scores, reverse=True)


def test_returned_citations_exist_in_canonical_corpus(client):
    corpus = json.loads(lightweight_retriever._KB_PATH.read_text(encoding="utf-8"))
    known_ids = {entry["id"] for entry in corpus}
    body = client.post(
        "/api/v1/rag/query", json={"query": "ISM-GBX-003 gear pitting", "top_k": 5}
    ).json()
    assert {source["doc_id"] for source in body["sources"]} <= known_ids
    assert all(source["source_file"] == "Ironside Manufacturing knowledge base" for source in body["sources"])


def test_top_k_is_respected_and_off_domain_abstains(client):
    one = client.post(
        "/api/v1/rag/query", json={"query": "ISM-CNC-001 bearing wear", "top_k": 1}
    ).json()
    assert len(one["sources"]) == 1
    off_domain = client.post(
        "/api/v1/rag/query", json={"query": "renaissance sonnet meter", "top_k": 5}
    ).json()
    assert off_domain["abstained"] is True
    assert off_domain["sources"] == []


def test_empty_and_corrupt_corpus_fail_safely(tmp_path, monkeypatch):
    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(lightweight_retriever, "_KB_PATH", empty)
    lightweight_retriever._entries.cache_clear()
    assert lightweight_retriever.search_ironside("ISM-CNC-001 bearing wear")["retrieved_passages"] == []

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(lightweight_retriever, "_KB_PATH", corrupt)
    lightweight_retriever._entries.cache_clear()
    with pytest.raises(json.JSONDecodeError):
        lightweight_retriever.search_ironside("ISM-CNC-001 bearing wear")
    lightweight_retriever._entries.cache_clear()
