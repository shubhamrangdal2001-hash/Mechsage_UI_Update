"""
backend/api/v1/websocket.py
-----------------------------
WebSocket endpoint: /ws/fleet

Clients connect here to receive real-time cycle_update and agent_alert
messages broadcast by the SimulationService.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.logging import get_logger
from backend.services.simulation_service import simulation_service

router = APIRouter(tags=["WebSocket"])
logger = get_logger(__name__)


@router.websocket("/ws/fleet")
async def fleet_websocket(ws: WebSocket) -> None:
    """
    WebSocket endpoint for real-time fleet telemetry push.

    Message types received by client:
      - cycle_update:       { type, payload: CyclePayload[], global_cycle }
      - agent_alert:        { type, payload: { asset_id, severity, rul, ... } }
      - simulation_status:  { type, payload: { status, engines } }

    Messages accepted from client:
      - { "action": "pause" }   — pause simulation
      - { "action": "resume" }  — resume simulation
      - { "action": "status" }  — get simulation status
    """
    await simulation_service.manager.connect(ws)

    # Send current snapshot immediately on connect
    try:
        snapshot = simulation_service.get_latest_snapshot()
        await ws.send_json({
            "type": "initial_snapshot",
            "payload": snapshot,
        })
    except Exception:
        pass

    try:
        while True:
            # Listen for control messages from the client
            data = await ws.receive_json()
            action = data.get("action")

            if action == "pause":
                simulation_service.pause()
                await ws.send_json({"type": "simulation_status", "payload": {"status": "paused"}})
            elif action == "resume":
                simulation_service.resume()
                await ws.send_json({"type": "simulation_status", "payload": {"status": "running"}})
            elif action == "status":
                await ws.send_json({
                    "type": "simulation_status",
                    "payload": {
                        "status": "running" if simulation_service.is_running else "stopped",
                        "connections": simulation_service.manager.connection_count(),
                        "cycle": simulation_service._cycle,
                    },
                })

    except WebSocketDisconnect:
        simulation_service.manager.disconnect(ws)
    except Exception as exc:
        logger.warning("WebSocket error", error=str(exc))
        simulation_service.manager.disconnect(ws)
