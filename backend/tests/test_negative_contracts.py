import pytest
from pydantic import ValidationError

from backend.schemas.inference import CyclePayload, InferRequest
from backend.schemas.rag import RAGQueryRequest
from backend.schemas.workorder import HITLActionRequest, WorkOrderCreate


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/v1/workorders", {}),
        ("/api/v1/workorders", {"asset_id": "ISM-CNC-001"}),
        (
            "/api/v1/workorders",
            {
                "asset_id": "ISM-CNC-001",
                "failure_mode": "bearing wear",
                "recommended_action": "inspect",
                "priority": "urgent",
            },
        ),
    ],
)
def test_invalid_work_order_payloads_return_structured_422(client, path, payload):
    response = client.post(path, json=payload)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "execute"},
        {"action": "approve", "proposed_start": "not-a-date"},
    ],
)
def test_invalid_hitl_payloads_are_rejected(client, payload):
    response = client.patch("/api/v1/workorders/999/action", json=payload)
    assert response.status_code == 422


def test_unknown_work_order_returns_404(client):
    assert client.get("/api/v1/workorders/999999").status_code == 404
    valid = client.patch("/api/v1/workorders/999999/action", json={"action": "reject"})
    assert valid.status_code == 404


@pytest.mark.parametrize("query", ["", "a", "  "])
def test_short_rag_queries_are_rejected(client, query):
    response = client.post("/api/v1/rag/query", json={"query": query})
    assert response.status_code == 422


def test_pydantic_models_reject_wrong_types_and_missing_fields():
    with pytest.raises(ValidationError):
        InferRequest.model_validate({})
    with pytest.raises(ValidationError):
        RAGQueryRequest.model_validate({"query": "ok", "top_k": 0})
    with pytest.raises(ValidationError):
        WorkOrderCreate.model_validate(
            {"asset_id": "x", "failure_mode": "x", "recommended_action": "x", "priority": "bad"}
        )
    with pytest.raises(ValidationError):
        HITLActionRequest.model_validate({"action": "approve", "proposed_start": "tomorrow"})
    with pytest.raises(ValidationError):
        CyclePayload.model_validate({"machine_id": "ISM-CNC-001"})


def test_resolved_work_order_cannot_be_decided_twice(client):
    created = client.post(
        "/api/v1/workorders",
        json={
            "asset_id": "ISM-CNC-001",
            "failure_mode": "bearing wear",
            "recommended_action": "Inspect and await human approval.",
        },
    ).json()
    first = client.patch(
        f"/api/v1/workorders/{created['id']}/action", json={"action": "reject", "notes": "Not required"}
    )
    second = client.patch(
        f"/api/v1/workorders/{created['id']}/action", json={"action": "approve"}
    )
    assert first.status_code == 200
    assert first.json()["status"] == "rejected"
    assert second.status_code == 409
    assert "only pending work orders" in second.json()["detail"]
