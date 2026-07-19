"""
backend/db/crud.py
-------------------
Database access functions (CRUD) for WorkOrder and HITLEvent.
All functions are synchronous — called from FastAPI async routes via
run_in_threadpool or direct sync DB dependency.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from backend.db.models import AuditLog, HITLEvent, WorkOrder
from backend.schemas.workorder import (
    HITLAction,
    HITLActionRequest,
    WorkOrderCreate,
    WorkOrderStatus,
)


class InvalidWorkOrderTransition(ValueError):
    """Raised when a resolved work order receives another HITL decision."""


# ── WorkOrder ─────────────────────────────────────────────────────────────────

def create_work_order(db: Session, data: WorkOrderCreate) -> WorkOrder:
    wo = WorkOrder(
        asset_id=data.asset_id,
        failure_mode=data.failure_mode,
        recommended_action=data.recommended_action,
        rul_estimate=data.rul_estimate,
        anomaly_score=data.anomaly_score,
        diagnosis_confidence=data.diagnosis_confidence,
        rag_context=data.rag_context,
        priority=data.priority.value,
        estimated_duration_hrs=data.estimated_duration_hrs,
        auto_generated=data.auto_generated,
        status=WorkOrderStatus.PENDING_APPROVAL.value,
    )
    wo.shap_top_features = data.shap_top_features
    wo.parts = data.parts
    wo.manual_refs = data.manual_refs

    db.add(wo)
    try:
        db.commit()
        db.refresh(wo)
    except Exception:
        db.rollback()
        raise
    return wo


def get_work_order(db: Session, wo_id: int) -> Optional[WorkOrder]:
    return db.get(WorkOrder, wo_id)


def list_work_orders(
    db: Session,
    status_filter: Optional[str] = None,
    asset_id_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[List[WorkOrder], int]:
    q = db.query(WorkOrder)
    if status_filter:
        q = q.filter(WorkOrder.status == status_filter)
    if asset_id_filter:
        q = q.filter(WorkOrder.asset_id == asset_id_filter)
    total = q.count()
    items = q.order_by(WorkOrder.created_at.desc()).offset(offset).limit(limit).all()
    return items, total


def apply_hitl_action(
    db: Session, wo_id: int, action_req: HITLActionRequest
) -> Optional[WorkOrder]:
    wo = get_work_order(db, wo_id)
    if wo is None:
        return None
    if wo.status != WorkOrderStatus.PENDING_APPROVAL.value:
        raise InvalidWorkOrderTransition(
            f"Work order {wo_id} is already {wo.status}; only pending work orders can be decided."
        )

    # Update work order status
    if action_req.action == HITLAction.APPROVE:
        wo.status = WorkOrderStatus.APPROVED.value
        wo.technician_id = action_req.technician_id
        wo.proposed_start = action_req.proposed_start
    else:
        wo.status = WorkOrderStatus.REJECTED.value
        wo.rejection_reason = action_req.notes

    wo.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    # Record HITL event
    event = HITLEvent(
        work_order_id=wo.id,
        action=action_req.action.value,
        technician_id=action_req.technician_id,
        notes=action_req.notes,
    )
    db.add(event)
    try:
        db.commit()
        db.refresh(wo)
    except Exception:
        db.rollback()
        raise
    return wo


def count_by_status(db: Session) -> dict[str, int]:
    from sqlalchemy import func
    rows = (
        db.query(WorkOrder.status, func.count(WorkOrder.id))
        .group_by(WorkOrder.status)
        .all()
    )
    return {status: count for status, count in rows}


# ── AuditLog ──────────────────────────────────────────────────────────────────

def log_event(
    db: Session,
    event_type: str,
    asset_id: Optional[str] = None,
    detail: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    entry = AuditLog(
        event_type=event_type,
        asset_id=asset_id,
        detail=detail,
        correlation_id=correlation_id,
    )
    db.add(entry)
    db.commit()
