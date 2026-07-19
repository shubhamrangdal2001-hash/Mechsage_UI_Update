"""
backend/api/v1/routers/workorders.py
--------------------------------------
CRUD + HITL endpoints for Work Orders.

Routes:
  GET    /api/v1/workorders          — List work orders (filterable)
  GET    /api/v1/workorders/{id}     — Get single work order
  POST   /api/v1/workorders          — Create a work order
  PATCH  /api/v1/workorders/{id}/action — Apply HITL approve/reject
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.core.security import verify_api_key
from backend.db import crud
from backend.db.session import get_db
from backend.schemas.workorder import (
    HITLActionRequest,
    WorkOrderCreate,
    WorkOrderListResponse,
    WorkOrderRead,
)

router = APIRouter(prefix="/workorders", tags=["Work Orders"])
logger = get_logger(__name__)


@router.get(
    "",
    response_model=WorkOrderListResponse,
    summary="List work orders",
    dependencies=[Depends(verify_api_key)],
)
def list_work_orders(
    status_filter: Optional[str] = Query(None, alias="status"),
    asset_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    items, total = crud.list_work_orders(
        db,
        status_filter=status_filter,
        asset_id_filter=asset_id,
        limit=limit,
        offset=offset,
    )
    counts = crud.count_by_status(db)

    return WorkOrderListResponse(
        items=[_to_read(wo) for wo in items],
        total=total,
        pending=counts.get("pending_approval", 0),
        approved=counts.get("approved", 0),
        rejected=counts.get("rejected", 0),
    )


@router.get(
    "/{wo_id}",
    response_model=WorkOrderRead,
    summary="Get a single work order",
    dependencies=[Depends(verify_api_key)],
)
def get_work_order(wo_id: int, db: Session = Depends(get_db)):
    wo = crud.get_work_order(db, wo_id)
    if wo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Work order {wo_id} not found.",
        )
    return _to_read(wo)


@router.post(
    "",
    response_model=WorkOrderRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a work order",
    dependencies=[Depends(verify_api_key)],
)
def create_work_order(data: WorkOrderCreate, db: Session = Depends(get_db)):
    wo = crud.create_work_order(db, data)
    logger.info("Work order created", wo_id=wo.id, asset_id=wo.asset_id)
    return _to_read(wo)


@router.patch(
    "/{wo_id}/action",
    response_model=WorkOrderRead,
    summary="Approve or reject a work order (HITL action)",
    dependencies=[Depends(verify_api_key)],
)
def hitl_action(
    wo_id: int,
    action_req: HITLActionRequest,
    db: Session = Depends(get_db),
):
    try:
        wo = crud.apply_hitl_action(db, wo_id, action_req)
    except crud.InvalidWorkOrderTransition as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    if wo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Work order {wo_id} not found.",
        )
    logger.info(
        "HITL action applied",
        wo_id=wo_id,
        action=action_req.action.value,
        new_status=wo.status,
    )
    return _to_read(wo)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_read(wo) -> WorkOrderRead:
    """Convert ORM model to Pydantic response schema."""
    return WorkOrderRead(
        id=wo.id,
        asset_id=wo.asset_id,
        failure_mode=wo.failure_mode,
        recommended_action=wo.recommended_action,
        rul_estimate=wo.rul_estimate,
        anomaly_score=wo.anomaly_score,
        shap_top_features=wo.shap_top_features,
        rag_context=wo.rag_context,
        manual_refs=wo.manual_refs,
        parts=wo.parts,
        priority=wo.priority,
        estimated_duration_hrs=wo.estimated_duration_hrs,
        diagnosis_confidence=wo.diagnosis_confidence,
        auto_generated=wo.auto_generated,
        status=wo.status,
        rejection_reason=wo.rejection_reason,
        proposed_start=wo.proposed_start,
        technician_id=wo.technician_id,
        created_at=wo.created_at,
        updated_at=wo.updated_at,
    )
