# backend/tests/test_inference.py

def test_on_demand_inference(client):
    """Test that manual on-demand inference endpoint works."""
    payload = {
        "dataset_id": "FD001",
        "cycle_override": 45
    }
    response = client.post("/api/v1/infer", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["machine_id"] == "ISM-CNC-001"
    assert data["dataset_variant"] == "FD001"
    assert "rul" in data
    assert "anomaly" in data
    assert data["rul"]["prediction_cycles"] == 85.5
    assert data["anomaly"]["score"] == 0.12

def test_inference_invalid_dataset(client):
    """Test that invalid dataset inputs are caught by validator."""
    payload = {
        "dataset_id": "FD099",
        "cycle_override": 10
    }
    response = client.post("/api/v1/infer", json=payload)
    assert response.status_code == 422

def test_fleet_snapshot(client):
    """Test that snapshot returns telemetry arrays."""
    response = client.get("/api/v1/infer/fleet/snapshot")
    assert response.status_code == 200
    data = response.json()
    assert "engines" in data
    assert "alert_count" in data
    assert "emergency_count" in data
