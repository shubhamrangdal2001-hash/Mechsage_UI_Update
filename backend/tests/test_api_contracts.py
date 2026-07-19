from backend.core.config import settings
from backend.core.security import verify_api_key


def test_health_and_correlation_id(client):
    response = client.get("/health", headers={"X-Correlation-ID": "qa-contract-123"})
    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "qa-contract-123"
    assert set(response.json()) == {
        "status", "inference_ready", "simulation_running", "ws_connections"
    }


def test_generated_correlation_id(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"]


def test_infer_rejects_missing_required_dataset(client):
    response = client.post("/api/v1/infer", json={})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "dataset_id"]


def test_history_rejects_invalid_dataset(client):
    response = client.get("/api/v1/infer/history/FD099")
    assert response.status_code == 422
    assert "Invalid dataset_id" in response.json()["detail"]


def test_pause_resume_contract(client):
    paused = client.post("/api/v1/infer/simulation/pause")
    resumed = client.post("/api/v1/infer/simulation/resume")
    assert paused.status_code == 200
    assert paused.json() == {"status": "paused"}
    assert resumed.status_code == 200
    assert resumed.json() == {"status": "running"}


def test_api_key_enforcement(client):
    client.app.dependency_overrides.pop(verify_api_key, None)
    original_required, original_key = settings.require_api_key, settings.api_key
    settings.require_api_key, settings.api_key = True, "unit-test-key"
    try:
        missing = client.get("/api/v1/workorders")
        wrong = client.get("/api/v1/workorders", headers={"X-API-Key": "wrong"})
        valid = client.get(
            "/api/v1/workorders", headers={"X-API-Key": "unit-test-key"}
        )
    finally:
        settings.require_api_key, settings.api_key = original_required, original_key
    assert missing.status_code == 401
    assert missing.headers["WWW-Authenticate"] == "ApiKey"
    assert wrong.status_code == 401
    assert valid.status_code == 200


def test_cors_preflight_for_frontend(client):
    response = client.options(
        "/api/v1/workorders",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_websocket_initial_snapshot_and_controls(client):
    with client.websocket_connect("/ws/fleet") as websocket:
        initial = websocket.receive_json()
        assert initial["type"] == "initial_snapshot"
        assert initial["payload"]["engines"][0]["machine_id"] == "ISM-CNC-001"

        websocket.send_json({"action": "pause"})
        assert websocket.receive_json() == {
            "type": "simulation_status", "payload": {"status": "paused"}
        }

        websocket.send_json({"unknown": "ignored"})
        websocket.send_json({"action": "status"})
        status = websocket.receive_json()
        assert status["type"] == "simulation_status"
        assert status["payload"]["status"] in {"running", "stopped"}

        websocket.send_json({"action": "resume"})
        assert websocket.receive_json() == {
            "type": "simulation_status", "payload": {"status": "running"}
        }
