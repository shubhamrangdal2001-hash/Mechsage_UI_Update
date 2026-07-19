# backend/tests/conftest.py
import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Setup system path to include repository root
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.main import app
from backend.db.session import Base, get_db
from backend.core.security import verify_api_key
from backend.services.inference_service import inference_service
from backend.services.simulation_service import simulation_service

# In-memory SQLite for testing
DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session():
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db_session, monkeypatch):
    # Override database dependency
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    
    # Bypass API key validation during unit tests
    def override_verify_api_key():
        return "test-api-key"
    
    app.dependency_overrides[verify_api_key] = override_verify_api_key

    # Mock InferenceService to prevent loading 8 heavy joblib models at test startup
    class MockInferenceService:
        def is_ready(self):
            return True
        def get_builder(self, ds):
            class MockBuilder:
                def update(self, row):
                    pass
            return MockBuilder()
        def predict(self, dataset_id, cycle, machine_id=None, cycle_override=None):
            return {
                "machine_id": machine_id or f"SIM-{dataset_id}",
                "dataset_variant": dataset_id,
                "cycle": cycle,
                "rul": {
                    "prediction_cycles": 85.5,
                    "alert": False,
                    "severity": "NORMAL",
                    "threshold_cycles": 30
                },
                "anomaly": {
                    "score": 0.12,
                    "alert": False,
                    "severity": "NORMAL",
                    "threshold": 0.52,
                    "model_type": "IsolationForest"
                },
                "trigger_agent": False,
                "agent_instruction": "Nominal"
            }
    
    async def no_op_startup():
        return None

    async def no_op_shutdown():
        return None

    # Temporarily stub heavyweight lifecycle and predictor behavior.
    original_predict = inference_service.predict
    original_ready = inference_service.is_ready
    original_inference_startup = inference_service.startup
    original_simulation_startup = simulation_service.startup
    original_simulation_shutdown = simulation_service.shutdown
    original_latest = simulation_service._latest.copy()
    
    inference_service.predict = MockInferenceService().predict
    inference_service.is_ready = MockInferenceService().is_ready
    inference_service.startup = no_op_startup
    simulation_service.startup = no_op_startup
    simulation_service.shutdown = no_op_shutdown
    simulation_service._latest = {
        "FD001": {
            "timestamp": "2026-07-19T00:00:00+00:00",
            "machine_id": "ISM-CNC-001",
            "dataset_variant": "FD001",
            "cycle": 45,
            "rul": {"prediction_cycles": 85.5, "alert": False, "severity": "NORMAL", "threshold_cycles": 30},
            "anomaly": {"score": 0.12, "alert": False, "severity": "NORMAL", "threshold": 0.52, "model_type": "IsolationForest"},
            "trigger_agent": False,
            "agent_instruction": "Nominal",
            "raw_features": {},
        }
    }

    # Never let unit tests create or mutate the configured application DB.
    monkeypatch.setattr("backend.main.create_tables", lambda: None)
    
    with TestClient(app) as c:
        yield c
        
    # Restore original methods
    inference_service.predict = original_predict
    inference_service.is_ready = original_ready
    inference_service.startup = original_inference_startup
    simulation_service.startup = original_simulation_startup
    simulation_service.shutdown = original_simulation_shutdown
    simulation_service._latest = original_latest
    app.dependency_overrides.clear()
