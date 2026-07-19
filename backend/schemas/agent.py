"""
backend/schemas/agent.py
------------------------
Pydantic v2 schemas for the LangGraph agentic pipeline request/response.
The AgentTrace exposes the full 5-node execution trace to the frontend.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentNodeTrace(BaseModel):
    """Output from one LangGraph node execution."""
    node: str = Field(..., description="Node name: supervisor | monitor | diagnostics | work_order | scheduling | human_review")
    status: str = Field(..., description="success | abstained | error")
    output: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[float] = None


class AgentRunRequest(BaseModel):
    """Request body to trigger a full agentic pipeline run."""
    asset_id: str = Field(..., description="Ironside unit identifier, e.g. ISM-CNC-001")
    dataset_id: str = Field(..., description="FD001 | FD002 | FD003 | FD004")
    cycle: Optional[int] = Field(None, description="Cycle number; uses latest if None")
    rul_estimate: Optional[float] = None
    anomaly_score: Optional[float] = None
    anomaly_flag: Optional[bool] = None
    raw_telemetry: Optional[Dict[str, float]] = None


class AgentRunResponse(BaseModel):
    """Full response from a LangGraph pipeline execution."""
    run_id: str
    asset_id: str
    started_at: datetime
    completed_at: datetime
    duration_ms: float
    final_status: str = Field(..., description="healthy | work_order_drafted | human_review | error")
    trace: List[AgentNodeTrace]
    work_order_id: Optional[int] = None
    work_order: Optional[Dict[str, Any]] = None
    schedule_proposal: Optional[str] = None
    approval_status: str = Field("N/A", description="pending_approval | needs_human | N/A")
    messages: List[str] = Field(default_factory=list)
