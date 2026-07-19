"""
backend/schemas/inference.py
-----------------------------
Pydantic v2 schemas for inference request/response payloads.
Mirrors the contract defined in inference/agent_feed.py exactly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Sub-schemas ───────────────────────────────────────────────────────────────

class RULResult(BaseModel):
    prediction_cycles: float = Field(..., description="Predicted remaining useful life in cycles")
    alert: bool = Field(..., description="True when RUL is below alert threshold")
    severity: str = Field(..., description="NORMAL | CRITICAL | EMERGENCY")
    threshold_cycles: int = Field(30, description="Alert threshold used")


class AnomalyResult(BaseModel):
    score: float = Field(..., description="Anomaly probability/score [0–1]")
    alert: bool = Field(..., description="True when score exceeds threshold")
    severity: str = Field(..., description="NORMAL | ANOMALY DETECTED")
    threshold: float = Field(..., description="Dataset-specific anomaly threshold")
    model_type: str = Field(..., description="IsolationForest | LightGBM")


# ── Core payload (matches agent_feed.build_payload output) ───────────────────

class CyclePayload(BaseModel):
    """Full structured prediction payload for one engine at one cycle."""
    timestamp: datetime
    machine_id: str
    dataset_variant: str
    cycle: int
    rul: RULResult
    anomaly: AnomalyResult
    trigger_agent: bool
    agent_instruction: str
    raw_features: Dict[str, float] = Field(default_factory=dict)


# ── Request / Response ────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    """On-demand inference for a specific dataset at a given cycle number."""
    dataset_id: str = Field(..., description="FD001 | FD002 | FD003 | FD004")
    machine_id: Optional[str] = None       # defaults to SIM-<dataset_id>
    cycle_override: Optional[int] = None   # if None, use current simulation cycle


class FleetSnapshot(BaseModel):
    """Latest prediction snapshot for all 4 engines."""
    engines: List[CyclePayload]
    alert_count: int
    emergency_count: int
    snapshot_at: datetime


# ── WebSocket message envelope ────────────────────────────────────────────────

class WSMessage(BaseModel):
    """WebSocket message pushed to connected clients."""
    type: str = Field(..., description="cycle_update | agent_alert | simulation_status")
    payload: Any
