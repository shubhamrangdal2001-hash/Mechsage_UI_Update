from pathlib import Path

from backend.main import app
from backend.schemas.inference import CyclePayload, FleetSnapshot
from backend.schemas.workorder import WorkOrderRead


ROOT = Path(__file__).resolve().parents[2]


def test_openapi_contains_public_rest_contracts():
    paths = app.openapi()["paths"]
    expected = {
        "/health",
        "/api/v1/infer",
        "/api/v1/infer/fleet/snapshot",
        "/api/v1/infer/history/{dataset_id}",
        "/api/v1/infer/simulation/pause",
        "/api/v1/infer/simulation/resume",
        "/api/v1/metrics",
        "/api/v1/rag/query",
        "/api/v1/agent/run",
        "/api/v1/workorders",
        "/api/v1/workorders/{wo_id}",
        "/api/v1/workorders/{wo_id}/action",
    }
    assert expected <= set(paths)


def test_typescript_declares_backend_response_fields():
    types = (ROOT / "frontend" / "types" / "index.ts").read_text(encoding="utf-8")
    for model in (CyclePayload, FleetSnapshot, WorkOrderRead):
        for field_name in model.model_fields:
            assert field_name in types, f"TypeScript contract missing {field_name}"


def test_typescript_websocket_message_names_match_backend():
    types = (ROOT / "frontend" / "types" / "index.ts").read_text(encoding="utf-8")
    for message_type in (
        "initial_snapshot", "cycle_update", "agent_alert", "simulation_status"
    ):
        assert f'"{message_type}"' in types
