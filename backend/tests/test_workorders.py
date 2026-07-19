# backend/tests/test_workorders.py

def test_work_order_workflow(client):
    """Test full workflow: Create -> Read -> List -> HITL Action."""
    
    # 1. Create a Work Order
    create_payload = {
        "asset_id": "ISM-CNC-001",
        "failure_mode": "CNC spindle bearing degradation",
        "recommended_action": "Execute the Ironside spindle-bearing inspection procedure.",
        "rul_estimate": 14.2,
        "anomaly_score": 0.725,
        "shap_top_features": ["s11", "s4", "s15"],
        "rag_context": "ANOMALY-BEARING-WEAR and ISM-CNC-001-PROFILE.",
        "manual_refs": ["ANOMALY-BEARING-WEAR", "ISM-CNC-001-PROFILE"],
        "parts": ["FAG B7013 spindle bearing set"],
        "priority": "high",
        "estimated_duration_hrs": 6.5,
        "diagnosis_confidence": 0.89,
        "auto_generated": True
    }
    
    response = client.post("/api/v1/workorders", json=create_payload)
    assert response.status_code == 201
    
    wo = response.json()
    assert wo["id"] is not None
    assert wo["asset_id"] == "ISM-CNC-001"
    assert wo["status"] == "pending_approval"
    
    wo_id = wo["id"]
    
    # 2. Retrieve the specific Work Order
    response = client.get(f"/api/v1/workorders/{wo_id}")
    assert response.status_code == 200
    assert response.json()["failure_mode"] == "CNC spindle bearing degradation"
    
    # 3. List and filter Work Orders
    response = client.get("/api/v1/workorders?status=pending_approval")
    assert response.status_code == 200
    list_data = response.json()
    assert list_data["total"] >= 1
    assert any(item["id"] == wo_id for item in list_data["items"])
    
    # 4. Perform HITL Action: Approve
    action_payload = {
        "action": "approve",
        "notes": "Spares in stock, dispatching technician.",
        "technician_id": "TECH-JOE",
        "proposed_start": "2026-07-20T08:00:00"
    }
    response = client.patch(f"/api/v1/workorders/{wo_id}/action", json=action_payload)
    assert response.status_code == 200
    
    updated_wo = response.json()
    assert updated_wo["status"] == "approved"
    assert updated_wo["technician_id"] == "TECH-JOE"
