"""
backend/api/v1/routers/metrics.py
------------------------------------
Fleet metrics endpoint — returns a structured JSON snapshot suitable
for driving frontend charts without needing the WebSocket.

Route:
  GET /api/v1/metrics   — Fleet health metrics snapshot
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from backend.core.security import verify_api_key
from backend.services.simulation_service import simulation_service

router = APIRouter(prefix="/metrics", tags=["Metrics"])

SEVERITY_ORDER = {"NORMAL": 0, "ANOMALY DETECTED": 1, "CRITICAL": 2, "EMERGENCY": 3}


@router.get(
    "",
    summary="Fleet health metrics snapshot",
    dependencies=[Depends(verify_api_key)],
)
async def get_metrics() -> Dict[str, Any]:
    """
    Returns a structured JSON metrics snapshot for all 4 engines.
    Designed to drive frontend charts without a WebSocket connection.
    """
    snapshot = simulation_service.get_latest_snapshot()
    engines: List[Dict] = snapshot.get("engines", [])

    engine_summaries = []
    for e in engines:
        if e.get("warming_up"):
            engine_summaries.append({
                "machine_id": e.get("machine_id"),
                "dataset_variant": e.get("dataset_variant"),
                "status": "warming_up",
                "warmup_progress": e.get("warmup_progress", 0),
            })
            continue

        rul = e.get("rul", {})
        anomaly = e.get("anomaly", {})
        engine_summaries.append({
            "machine_id": e.get("machine_id"),
            "dataset_variant": e.get("dataset_variant"),
            "cycle": e.get("cycle"),
            "rul_cycles": rul.get("prediction_cycles"),
            "rul_severity": rul.get("severity"),
            "rul_alert": rul.get("alert"),
            "anomaly_score": anomaly.get("score"),
            "anomaly_severity": anomaly.get("severity"),
            "anomaly_alert": anomaly.get("alert"),
            "trigger_agent": e.get("trigger_agent"),
            "timestamp": e.get("timestamp"),
        })

    # Fleet-wide stats
    active_engines = [e for e in engine_summaries if e.get("status") != "warming_up"]
    alert_engines = [e for e in active_engines if e.get("trigger_agent")]
    emergency_engines = [e for e in active_engines if e.get("rul_severity") == "EMERGENCY"]

    worst_rul = None
    if active_engines:
        ruls = [e["rul_cycles"] for e in active_engines if e.get("rul_cycles") is not None]
        worst_rul = min(ruls) if ruls else None

    return {
        "snapshot_at": datetime.now(tz=timezone.utc).isoformat(),
        "simulation": {
            "running": simulation_service.is_running,
            "global_cycle": simulation_service._cycle,
            "connections": simulation_service.manager.connection_count(),
        },
        "fleet": {
            "total_engines": len(engine_summaries),
            "active_engines": len(active_engines),
            "alert_count": len(alert_engines),
            "emergency_count": len(emergency_engines),
            "worst_rul_cycles": worst_rul,
        },
        "engines": engine_summaries,
    }
