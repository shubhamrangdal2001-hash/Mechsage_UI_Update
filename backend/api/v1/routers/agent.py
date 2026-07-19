"""
backend/api/v1/routers/agent.py
---------------------------------
Trigger the full LangGraph 5-node agentic pipeline.

Route:
  POST /api/v1/agent/run   — Run supervisor → monitor → diagnostics → work_order → scheduling
"""

from __future__ import annotations

import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from backend.core.logging import get_logger
from backend.core.security import verify_api_key
from backend.schemas.agent import AgentNodeTrace, AgentRunRequest, AgentRunResponse
from backend.services.simulation_service import simulation_service

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

router = APIRouter(prefix="/agent", tags=["Agentic Pipeline"])
logger = get_logger(__name__)


@router.post(
    "/run",
    response_model=AgentRunResponse,
    summary="Run full LangGraph agentic pipeline",
    dependencies=[Depends(verify_api_key)],
)
async def run_agent(req: AgentRunRequest):
    """
    Trigger the MechSage 5-node LangGraph pipeline for a given asset.
    Returns the full node-by-node execution trace plus any work order generated.
    """
    from dev.agentic.graph import mechsage_graph

    run_id = str(uuid.uuid4())
    started_at = datetime.now(tz=timezone.utc)
    t0 = time.perf_counter()

    # Build initial state from the request + latest simulation data
    latest = simulation_service.get_latest_snapshot()
    engine_data = next(
        (e for e in latest.get("engines", [])
         if e.get("machine_id") == req.asset_id),
        None,
    )

    rul_estimate = req.rul_estimate
    anomaly_score = req.anomaly_score
    anomaly_flag = req.anomaly_flag

    if engine_data:
        rul_estimate = rul_estimate or engine_data.get("rul", {}).get("prediction_cycles")
        anomaly_score = anomaly_score or engine_data.get("anomaly", {}).get("score")
        anomaly_flag = anomaly_flag if anomaly_flag is not None else engine_data.get("trigger_agent", False)

    initial_state = {
        "asset_id": req.asset_id,
        "asset_type": "ironside",
        "status": "initialised",
        "error": None,
        "messages": [],
        "raw_telemetry": req.raw_telemetry or {},
        "rul_estimate": rul_estimate,
        "anomaly_flag": anomaly_flag or False,
        "anomaly_score": anomaly_score or 0.0,
        "degrading_sensors": [],
        "fault_hypothesis": "",
        "retrieved_passages": [],
        "diagnosis": "",
        "confidence": 0.0,
        "citation": "",
        "work_order": {},
        "schedule_proposal": "",
        "approval_status": "pending",
    }

    try:
        final_state = mechsage_graph.invoke(initial_state)
    except Exception as exc:
        logger.error("Agent pipeline failed", error=str(exc), run_id=run_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent pipeline error: {str(exc)}",
        )

    duration_ms = (time.perf_counter() - t0) * 1000
    completed_at = datetime.now(tz=timezone.utc)

    # Parse execution trace from messages
    messages = final_state.get("messages", [])
    trace = _build_trace(final_state, messages)

    final_status = final_state.get("status", "unknown")
    work_order = final_state.get("work_order") or {}

    logger.info(
        "Agent run complete",
        run_id=run_id,
        asset_id=req.asset_id,
        final_status=final_status,
        duration_ms=round(duration_ms, 1),
    )

    return AgentRunResponse(
        run_id=run_id,
        asset_id=req.asset_id,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=round(duration_ms, 1),
        final_status=final_status,
        trace=trace,
        work_order=work_order if work_order else None,
        schedule_proposal=final_state.get("schedule_proposal"),
        approval_status=final_state.get("approval_status", "N/A"),
        messages=messages,
    )


def _build_trace(final_state: dict, messages: list[str]) -> list[AgentNodeTrace]:
    """
    Build a human-readable trace from the final LangGraph state.
    LangGraph doesn't expose step-by-step timing natively, so we
    infer nodes from the state fields that were populated.
    """
    nodes_in_order = [
        ("supervisor",  {"asset_id": final_state.get("asset_id"), "asset_type": final_state.get("asset_type")}),
        ("monitor",     {"rul_estimate": final_state.get("rul_estimate"), "anomaly_flag": final_state.get("anomaly_flag"), "anomaly_score": final_state.get("anomaly_score"), "degrading_sensors": final_state.get("degrading_sensors")}),
    ]

    status = final_state.get("status", "")

    if status in ("abstain", "human_review"):
        nodes_in_order.append(
            ("diagnostics", {"diagnosis": "unresolved", "confidence": final_state.get("confidence"), "status": "abstained"})
        )
        nodes_in_order.append(
            ("human_review", {"approval_status": final_state.get("approval_status"), "status": "needs_human"})
        )
    elif final_state.get("diagnosis"):
        nodes_in_order.append(
            ("diagnostics", {"diagnosis": final_state.get("diagnosis"), "confidence": final_state.get("confidence"), "citation": final_state.get("citation")})
        )
        if final_state.get("work_order"):
            nodes_in_order.append(
                ("work_order", {"work_order": final_state.get("work_order")})
            )
        if final_state.get("schedule_proposal"):
            nodes_in_order.append(
                ("scheduling", {"schedule_proposal": final_state.get("schedule_proposal"), "approval_status": final_state.get("approval_status")})
            )

    trace = []
    for node_name, output in nodes_in_order:
        node_status = "abstained" if (node_name in ("diagnostics", "human_review") and status in ("abstain", "human_review")) else "success"
        trace.append(AgentNodeTrace(node=node_name, status=node_status, output=output))

    return trace
