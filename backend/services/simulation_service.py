"""
backend/services/simulation_service.py
----------------------------------------
Background asyncio service that runs the 4-engine simulation loop
and broadcasts prediction payloads to all connected WebSocket clients.

Architecture:
  - One asyncio.Task runs the simulation loop continuously
  - A set of active WebSocket connections is maintained
  - Each cycle payload is broadcast to ALL connected clients
  - Supports pause/resume from the frontend
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.services.inference_service import inference_service

logger = get_logger(__name__)

DATASETS = ("FD001", "FD002", "FD003", "FD004")
IRONSIDE_ASSETS = {
    "FD001": "ISM-CNC-001",
    "FD002": "ISM-HYD-002",
    "FD003": "ISM-GBX-003",
    "FD004": "ISM-CMR-004",
}
IRONSIDE_EMERGENCY_ACTIONS = {
    "FD001": "CNC MACHINING CENTER FAILURE RISK. Initiate controlled stop and retrieve the ISM-CNC-001 spindle-bearing maintenance procedure.",
    "FD002": "HYDRAULIC PRESS FAILURE RISK. Isolate pressure, apply LOTO, and retrieve the ISM-HYD-002 hydraulic-system maintenance procedure.",
    "FD003": "HELICAL GEARBOX FAILURE RISK. Reduce drive load and retrieve the ISM-GBX-003 gearbox inspection procedure.",
    "FD004": "ROTARY SCREW COMPRESSOR FAILURE IMMINENT. Initiate controlled shutdown and retrieve the ISM-CMR-004 air-end bearing procedure.",
}
IRONSIDE_WORK_ORDER_META = {
    "FD001": {
        "failure_mode": "CNC spindle or process anomaly detected",
        "manual_refs": ["ISM-CNC-001-PROFILE", "ANOMALY-BEARING-WEAR"],
        "parts": ["Inspect against ISM-CNC-001 profile before selecting spares"],
        "duration": 6.0,
    },
    "FD002": {
        "failure_mode": "Hydraulic press telemetry anomaly detected",
        "manual_refs": ["ISM-HYD-002-PROFILE", "ANOMALY-CAVITATION"],
        "parts": ["Inspect hydraulic circuit before selecting seals or pump parts"],
        "duration": 4.0,
    },
    "FD003": {
        "failure_mode": "Helical gearbox vibration anomaly detected",
        "manual_refs": ["ISM-GBX-003-PROFILE", "ANOMALY-GEAR-PITTING"],
        "parts": ["Confirm gearbox component applicability after vibration inspection"],
        "duration": 8.0,
    },
    "FD004": {
        "failure_mode": "Rotary screw compressor failure risk detected",
        "manual_refs": ["ISM-CMR-004-PROFILE", "ANOMALY-BEARING-WEAR"],
        "parts": ["Confirm air-end bearing assembly applicability after inspection"],
        "duration": 8.0,
    },
}


class ConnectionManager:
    """Manages the set of active WebSocket connections."""

    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.add(ws)
        logger.info("WebSocket client connected", total=len(self._active))

    def disconnect(self, ws: WebSocket) -> None:
        self._active.discard(ws)
        logger.info("WebSocket client disconnected", total=len(self._active))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast JSON message to all active clients, dropping dead connections."""
        if not self._active:
            return
        data = json.dumps(message, default=str)
        dead: Set[WebSocket] = set()
        for ws in self._active.copy():
            try:
                await asyncio.wait_for(ws.send_text(data), timeout=0.5)
            except Exception:
                dead.add(ws)
        self._active -= dead

    def connection_count(self) -> int:
        return len(self._active)


class SimulationService:
    """
    Manages the background simulation loop.
    Engines auto-reset when they fail, giving a continuous stream.
    """

    def __init__(self) -> None:
        self.manager = ConnectionManager()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._paused = False
        self._cycle: int = 0
        # Latest snapshot per engine (dataset_id → payload dict)
        self._latest: Dict[str, Dict[str, Any]] = {}
        # History of payloads per engine (dataset_id -> list of payload dicts)
        self.history: Dict[str, List[Dict[str, Any]]] = {ds: [] for ds in DATASETS}
        self._drafted_active_alerts: Set[str] = set()
        self._draft_lock = threading.Lock()

    async def startup(self) -> None:
        if settings.simulation_auto_start:
            await self.start()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._simulation_loop())
        logger.info("Simulation loop started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Simulation loop stopped")

    def pause(self) -> None:
        self._paused = True
        logger.info("Simulation paused")

    def resume(self) -> None:
        self._paused = False
        logger.info("Simulation resumed")

    def get_latest_snapshot(self) -> Dict[str, Any]:
        """Return the most recent payload for every engine."""
        engines = list(self._latest.values())
        alert_count = sum(1 for e in engines if e.get("trigger_agent"))
        emergency_count = sum(
            1 for e in engines if e.get("rul", {}).get("severity") == "EMERGENCY"
        )
        return {
            "engines": engines,
            "alert_count": alert_count,
            "emergency_count": emergency_count,
            "snapshot_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    @property
    def is_running(self) -> bool:
        return self._running and bool(self._task) and not self._task.done()

    # ── Private ───────────────────────────────────────────────────────────────

    async def _simulation_loop(self) -> None:
        """
        Core simulation loop.
        Creates engines and builders, then loops indefinitely.
        Engines auto-reset on failure for a continuous stream.
        """
        from inference.synthetic_generator import create_engines

        engines = create_engines(
            datasets=list(DATASETS),
            max_cycles=settings.simulation_max_cycles,
            seed=settings.simulation_seed,
        )

        logger.info("Engines created", count=len(engines))

        # Broadcast startup event
        await self.manager.broadcast({
            "type": "simulation_status",
            "payload": {"status": "started", "engines": list(DATASETS)},
        })

        while self._running:
            if self._paused:
                await asyncio.sleep(0.1)
                continue

            self._cycle += 1
            cycle_payloads = []

            for ds in DATASETS:
                engine = engines[ds]

                # Auto-reset failed engine
                if engine.is_failed:
                    engine.reset()
                    self.history[ds] = []
                    # Also reset the streaming feature builder
                    builder = inference_service.get_builder(ds)
                    if builder:
                        builder.reset()

                raw_row = engine.next_cycle()

                payload = inference_service.predict(
                    dataset_id=ds,
                    raw_row=raw_row,
                    cycle=engine.current_cycle,
                    machine_id=IRONSIDE_ASSETS[ds],
                )

                if payload is None:
                    # Still warming up — broadcast progress
                    builder = inference_service.get_builder(ds)
                    warmup_payload = {
                        "machine_id": IRONSIDE_ASSETS[ds],
                        "dataset_variant": ds,
                        "cycle": engine.current_cycle,
                        "warming_up": True,
                        "warmup_progress": (
                            builder.n_cycles / builder.warmup_cycles
                            if builder else 0
                        ),
                    }
                    cycle_payloads.append(warmup_payload)
                    continue

                new_alert = False
                if payload.get("trigger_agent"):
                    payload["agent_instruction"] = IRONSIDE_EMERGENCY_ACTIONS[ds]
                    was_active = IRONSIDE_ASSETS[ds] in self._drafted_active_alerts
                    self._ensure_pending_work_order(ds, payload)
                    new_alert = not was_active and IRONSIDE_ASSETS[ds] in self._drafted_active_alerts
                else:
                    self._drafted_active_alerts.discard(IRONSIDE_ASSETS[ds])

                self._latest[ds] = payload
                self.history[ds].append(payload)
                if len(self.history[ds]) > 300:
                    self.history[ds].pop(0)
                cycle_payloads.append(payload)

                # Broadcast agent alert separately for high-priority events
                if new_alert:
                    await self.manager.broadcast({
                        "type": "agent_alert",
                        "payload": {
                            "asset_id": payload["machine_id"],
                            "severity": payload["rul"]["severity"],
                            "rul": payload["rul"]["prediction_cycles"],
                            "anomaly_score": payload["anomaly"]["score"],
                            "timestamp": payload["timestamp"],
                        },
                    })

            # Broadcast full cycle update
            if cycle_payloads:
                await self.manager.broadcast({
                    "type": "cycle_update",
                    "payload": cycle_payloads,
                    "global_cycle": self._cycle,
                })

            await asyncio.sleep(settings.simulation_sleep_seconds)

    def _ensure_pending_work_order(self, dataset_id: str, payload: Dict[str, Any]) -> None:
        """Persist one HITL draft when a new live alert becomes active."""
        asset_id = IRONSIDE_ASSETS[dataset_id]
        with self._draft_lock:
            self._ensure_pending_work_order_locked(dataset_id, payload, asset_id)

    def _ensure_pending_work_order_locked(
        self, dataset_id: str, payload: Dict[str, Any], asset_id: str
    ) -> None:
        """Serialized implementation preventing duplicate concurrent drafts."""
        if asset_id in self._drafted_active_alerts:
            return

        from backend.db import crud
        from backend.db.models import WorkOrder
        from backend.db.session import SessionLocal
        from backend.schemas.workorder import Priority, WorkOrderCreate

        db = SessionLocal()
        try:
            existing = (
                db.query(WorkOrder)
                .filter(
                    WorkOrder.asset_id == asset_id,
                    WorkOrder.status == "pending_approval",
                )
                .first()
            )
            if existing is None:
                meta = IRONSIDE_WORK_ORDER_META[dataset_id]
                rul = float(payload.get("rul", {}).get("prediction_cycles", 0.0))
                anomaly_score = float(payload.get("anomaly", {}).get("score", 0.0))
                priority = Priority.CRITICAL if rul < 20 else Priority.HIGH
                work_order = crud.create_work_order(
                    db,
                    WorkOrderCreate(
                        asset_id=asset_id,
                        failure_mode=meta["failure_mode"],
                        recommended_action=payload["agent_instruction"],
                        rul_estimate=rul,
                        anomaly_score=anomaly_score,
                        rag_context=(
                            f"Live Ironside telemetry alert for {asset_id}: "
                            f"RUL={rul:.1f} cycles, anomaly score={anomaly_score:.3f}."
                        ),
                        manual_refs=meta["manual_refs"],
                        parts=meta["parts"],
                        priority=priority,
                        estimated_duration_hrs=meta["duration"],
                        diagnosis_confidence=0.90,
                        auto_generated=True,
                    ),
                )
                logger.info(
                    "Live alert work order drafted",
                    wo_id=work_order.id,
                    asset_id=asset_id,
                )
            self._drafted_active_alerts.add(asset_id)
        except Exception as exc:
            db.rollback()
            logger.error("Failed to draft live work order", asset_id=asset_id, error=str(exc))
        finally:
            db.close()

    async def shutdown(self) -> None:
        await self.stop()


# Global singleton
simulation_service = SimulationService()
