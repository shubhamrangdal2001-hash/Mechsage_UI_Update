"""
backend/api/v1/routers/inference.py
-------------------------------------
REST endpoints for on-demand inference and fleet snapshot.

Routes:
  POST /api/v1/infer          — Run inference for a specific dataset/cycle
  GET  /api/v1/fleet/snapshot — Get latest snapshot for all 4 engines
  POST /api/v1/simulation/pause   — Pause the simulation loop
  POST /api/v1/simulation/resume  — Resume the simulation loop
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from backend.core.security import verify_api_key
from backend.schemas.inference import FleetSnapshot, InferRequest
from backend.services.inference_service import inference_service
from backend.services.simulation_service import simulation_service

router = APIRouter(prefix="/infer", tags=["Inference"])


@router.post(
    "",
    summary="On-demand cycle inference",
    dependencies=[Depends(verify_api_key)],
)
async def infer(req: InferRequest):
    """
    Run RUL + Anomaly inference for a specific dataset.
    Returns the latest available payload from the simulation for that dataset,
    or triggers a fresh prediction if a cycle_override is provided.
    """
    if not inference_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service is not ready yet. Please wait for model loading.",
        )

    ds = req.dataset_id.upper()
    if ds not in ("FD001", "FD002", "FD003", "FD004"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid dataset_id '{ds}'. Must be FD001–FD004.",
        )

    # Return latest simulation snapshot for this engine
    snapshot = simulation_service.get_latest_snapshot()
    for engine_payload in snapshot["engines"]:
        if engine_payload.get("dataset_variant") == ds:
            return engine_payload

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No prediction available yet for {ds}. Simulation may still be warming up.",
    )


@router.get(
    "/fleet/snapshot",
    response_model=FleetSnapshot,
    summary="Latest fleet snapshot",
    dependencies=[Depends(verify_api_key)],
)
async def fleet_snapshot():
    """Return the most recent prediction for all 4 engines."""
    if not inference_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not ready.",
        )

    raw = simulation_service.get_latest_snapshot()
    engines = raw.get("engines", [])

    return FleetSnapshot(
        engines=engines,
        alert_count=raw["alert_count"],
        emergency_count=raw["emergency_count"],
        snapshot_at=datetime.now(tz=timezone.utc),
    )


@router.post(
    "/simulation/pause",
    summary="Pause the simulation loop",
    dependencies=[Depends(verify_api_key)],
)
async def pause_simulation():
    simulation_service.pause()
    return {"status": "paused"}


@router.post(
    "/simulation/resume",
    summary="Resume the simulation loop",
    dependencies=[Depends(verify_api_key)],
)
async def resume_simulation():
    simulation_service.resume()
    return {"status": "running"}


@router.get(
    "/history/{dataset_id}",
    summary="Get engine prediction history",
    dependencies=[Depends(verify_api_key)],
)
async def get_history(dataset_id: str):
    """Return prediction history list for a specific engine."""
    ds = dataset_id.upper()
    if ds not in ("FD001", "FD002", "FD003", "FD004"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid dataset_id '{ds}'",
        )
    return simulation_service.history.get(ds, [])
