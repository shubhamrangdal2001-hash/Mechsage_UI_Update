# backend/tests/test_rag_guardrails.py


def test_rag_query_nominal(client):
    """Ironside queries return grounded corpus citations."""
    response = client.post(
        "/api/v1/rag/query",
        json={
            "query": "What is the bearing wear procedure for ISM-CNC-001 when s11 vibration rises?",
            "top_k": 3,
            "dataset_context": "ISM-CNC-001",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["abstained"] is False
    assert data["sources"]
    assert all(source["source_file"] == "Ironside Manufacturing knowledge base" for source in data["sources"])


def test_rag_query_abstained(client):
    """Off-domain queries trigger the retrieval guardrail."""
    response = client.post(
        "/api/v1/rag/query",
        json={"query": "medieval poetry rhyme scheme", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["abstained"] is True
    assert data["sources"] == []
