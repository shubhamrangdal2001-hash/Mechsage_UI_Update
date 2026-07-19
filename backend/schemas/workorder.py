"""
backend/schemas/workorder.py
-----------------------------
Pydantic v2 schemas for Work Orders and Human-in-the-Loop (HITL) actions.
Mirrors the work_order contract defined in 03_architecture.md §4.3.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WorkOrderStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class HITLAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


# ── Request / Response ────────────────────────────────────────────────────────

class WorkOrderCreate(BaseModel):
    """Schema for creating a work order (typically from the agentic pipeline)."""
    asset_id: str
    failure_mode: str
    recommended_action: str
    rul_estimate: Optional[float] = None
    anomaly_score: Optional[float] = None
    shap_top_features: List[str] = Field(default_factory=list)
    rag_context: Optional[str] = None
    manual_refs: List[str] = Field(default_factory=list)
    parts: List[str] = Field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    estimated_duration_hrs: float = 1.0
    diagnosis_confidence: float = 0.0
    auto_generated: bool = True


class WorkOrderRead(BaseModel):
    """Full work order with DB-assigned fields."""
    id: int
    asset_id: str
    failure_mode: str
    recommended_action: str
    rul_estimate: Optional[float]
    anomaly_score: Optional[float]
    shap_top_features: List[str]
    rag_context: Optional[str]
    manual_refs: List[str]
    parts: List[str]
    priority: Priority
    estimated_duration_hrs: float
    diagnosis_confidence: float
    auto_generated: bool
    status: WorkOrderStatus
    rejection_reason: Optional[str]
    proposed_start: Optional[datetime]
    technician_id: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class HITLActionRequest(BaseModel):
    """Body for approve/reject action on a work order."""
    action: HITLAction
    notes: Optional[str] = None
    technician_id: Optional[str] = None
    proposed_start: Optional[datetime] = None


class WorkOrderListResponse(BaseModel):
    items: List[WorkOrderRead]
    total: int
    pending: int
    approved: int
    rejected: int
