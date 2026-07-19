from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db import crud
from backend.db.models import Base, HITLEvent, WorkOrder
from backend.schemas.workorder import HITLActionRequest, WorkOrderCreate
from backend.services.simulation_service import IRONSIDE_ASSETS, simulation_service


def _alert_payload(asset_id="ISM-CNC-001"):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "machine_id": asset_id,
        "agent_instruction": "Inspect the Ironside unit and keep the draft pending.",
        "rul": {"prediction_cycles": 12.0},
        "anomaly": {"score": 0.91},
    }


def test_rejection_and_approval_create_hitl_events(db_session):
    first = crud.create_work_order(
        db_session,
        WorkOrderCreate(asset_id="ISM-HYD-002", failure_mode="cavitation", recommended_action="inspect"),
    )
    rejected = crud.apply_hitl_action(
        db_session, first.id, HITLActionRequest(action="reject", notes="False positive")
    )
    assert rejected.status == "rejected"
    assert rejected.rejection_reason == "False positive"

    second = crud.create_work_order(
        db_session,
        WorkOrderCreate(asset_id="ISM-GBX-003", failure_mode="gear pitting", recommended_action="inspect"),
    )
    start = datetime(2026, 7, 21, 8, 30)
    approved = crud.apply_hitl_action(
        db_session,
        second.id,
        HITLActionRequest(action="approve", technician_id="TECH-42", proposed_start=start),
    )
    assert approved.status == "approved"
    assert approved.technician_id == "TECH-42"
    assert approved.proposed_start == start
    assert db_session.query(HITLEvent).filter_by(work_order_id=first.id).count() == 1
    assert db_session.query(HITLEvent).filter_by(work_order_id=second.id).count() == 1


def test_filters_pagination_and_counts(client):
    for asset in ("ISM-CNC-001", "ISM-HYD-002", "ISM-GBX-003"):
        response = client.post(
            "/api/v1/workorders",
            json={"asset_id": asset, "failure_mode": "qa", "recommended_action": "review"},
        )
        assert response.status_code == 201
    filtered = client.get("/api/v1/workorders?asset_id=ISM-HYD-002&limit=1&offset=0").json()
    assert filtered["total"] == 1
    assert len(filtered["items"]) == 1
    assert filtered["items"][0]["asset_id"] == "ISM-HYD-002"
    assert filtered["pending"] == 3


def test_create_rolls_back_when_commit_fails(monkeypatch, db_session):
    rolled_back = False
    original_rollback = db_session.rollback

    def tracking_rollback():
        nonlocal rolled_back
        rolled_back = True
        return original_rollback()

    monkeypatch.setattr(db_session, "commit", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    monkeypatch.setattr(db_session, "rollback", tracking_rollback)
    try:
        crud.create_work_order(
            db_session,
            WorkOrderCreate(asset_id="ISM-CMR-004", failure_mode="fatigue", recommended_action="inspect"),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("commit failure must propagate")
    assert rolled_back


def test_live_alert_dedup_concurrency_and_reactivation(tmp_path, monkeypatch):
    db_path = tmp_path / "alerts.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr("backend.db.session.SessionLocal", sessions)
    asset = IRONSIDE_ASSETS["FD001"]
    simulation_service._drafted_active_alerts.discard(asset)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: simulation_service._ensure_pending_work_order("FD001", _alert_payload(asset)), range(8)))

    with sessions() as db:
        drafts = db.query(WorkOrder).filter_by(asset_id=asset, status="pending_approval").all()
        assert len(drafts) == 1
        crud.apply_hitl_action(db, drafts[0].id, HITLActionRequest(action="reject", notes="cleared"))

    # A cleared alert removes the in-memory transition latch; reactivation may draft again.
    simulation_service._drafted_active_alerts.discard(asset)
    simulation_service._ensure_pending_work_order("FD001", _alert_payload(asset))
    with sessions() as db:
        assert db.query(WorkOrder).filter_by(asset_id=asset).count() == 2
        assert db.query(WorkOrder).filter_by(asset_id=asset, status="pending_approval").count() == 1
    simulation_service._drafted_active_alerts.discard(asset)
