"""Real artifact integration tests; external services are not involved."""

import math
import warnings

import pytest

from inference.agent_feed import build_payload
from inference.feature_builder import StreamingFeatureBuilder
from inference.predictor import DATASETS, SERVE_MODELS_DIR, load_all_models, predict_cycle
from inference.synthetic_generator import SyntheticEngine


ASSET_IDS = {
    "FD001": "ISM-CNC-001",
    "FD002": "ISM-HYD-002",
    "FD003": "ISM-GBX-003",
    "FD004": "ISM-CMR-004",
}


@pytest.fixture(scope="module")
def real_models():
    return load_all_models()


def test_all_eight_joblib_artifacts_exist_and_load(real_models):
    assert set(real_models) == set(DATASETS)
    for dataset_id in DATASETS:
        for kind, prefix in (("rul", "RUL"), ("anomaly", "Anomaly")):
            path = SERVE_MODELS_DIR / f"{prefix}_{dataset_id}_model.joblib"
            assert path.is_file() and path.stat().st_size > 0
            model = real_models[dataset_id][kind]
            assert callable(getattr(model, "predict", None))
        anomaly = real_models[dataset_id]["anomaly"]
        assert callable(getattr(anomaly, "predict_proba", None)) or callable(
            getattr(anomaly, "decision_function", None)
        )


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_synthetic_cycles_use_real_scaler_and_models(dataset_id, real_models):
    asset_id = ASSET_IDS[dataset_id]
    engine = SyntheticEngine(
        unit_id=asset_id, dataset=dataset_id, max_cycles=200, seed=100
    )
    builder = StreamingFeatureBuilder(dataset_id=dataset_id, warmup_cycles=20)

    feature_row = None
    raw_row = None
    for _ in range(20):
        raw_row = engine.next_cycle()
        feature_row = builder.add_cycle(raw_row)
        if engine.current_cycle < 20:
            assert feature_row is None

    assert feature_row is not None
    assert list(feature_row.columns) == builder._feature_cols
    expected_features = getattr(real_models[dataset_id]["rul"], "n_features_in_", None)
    if expected_features is not None:
        assert feature_row.shape[1] == expected_features

    prediction = predict_cycle(
        dataset_id=dataset_id,
        feature_row=feature_row,
        models=real_models,
        cycle=engine.current_cycle,
        machine_id=asset_id,
    )
    payload = build_payload(prediction, raw_row)

    assert payload["machine_id"] == asset_id
    assert "SIM-FD" not in payload["machine_id"]
    assert math.isfinite(payload["rul"]["prediction_cycles"])
    assert payload["rul"]["prediction_cycles"] >= 0
    assert math.isfinite(payload["anomaly"]["score"])
    assert 0 <= payload["anomaly"]["score"] <= 1
    assert payload["anomaly"]["model_type"] in {"IsolationForest", "LightGBM"}
    assert payload["trigger_agent"] == (
        payload["rul"]["alert"] or payload["anomaly"]["alert"]
    )
    assert payload["rul"]["severity"] in {"NORMAL", "CRITICAL", "EMERGENCY"}
    assert payload["anomaly"]["severity"] in {"NORMAL", "ANOMALY DETECTED"}

def test_fixed_seed_produces_identical_sensor_stream():
    for dataset_id in DATASETS:
        left = SyntheticEngine("left", dataset_id, seed=77)
        right = SyntheticEngine("right", dataset_id, seed=77)
        for _ in range(40):
            assert left.next_cycle().equals(right.next_cycle())


@pytest.mark.production_gate
def test_joblib_runtime_version_is_compatible():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_all_models()
    incompatibilities = [
        str(item.message) for item in caught if "InconsistentVersionWarning" in type(item.message).__name__
    ]
    assert not incompatibilities, "\n".join(incompatibilities)


@pytest.mark.production_gate
@pytest.mark.parametrize("dataset_id", DATASETS)
def test_real_model_rul_stays_within_simulation_regression_bound(dataset_id, real_models):
    engine = SyntheticEngine(ASSET_IDS[dataset_id], dataset_id, max_cycles=300, seed=123)
    builder = StreamingFeatureBuilder(dataset_id, warmup_cycles=20)
    row = None
    raw = None
    for _ in range(20):
        raw = engine.next_cycle()
        row = builder.add_cycle(raw)
    prediction = predict_cycle(dataset_id, row, real_models, 20, ASSET_IDS[dataset_id])
    assert 0 <= prediction["rul_prediction"] <= 300


@pytest.mark.production_gate
@pytest.mark.parametrize("dataset_id", DATASETS)
def test_real_predictions_change_across_degradation_cycles(dataset_id, real_models):
    engine = SyntheticEngine(ASSET_IDS[dataset_id], dataset_id, max_cycles=200, seed=100)
    builder = StreamingFeatureBuilder(dataset_id, warmup_cycles=20)
    predictions = []
    for cycle in range(1, 41):
        raw = engine.next_cycle()
        row = builder.add_cycle(raw)
        if cycle in {20, 40}:
            predictions.append(
                predict_cycle(dataset_id, row, real_models, cycle, ASSET_IDS[dataset_id])
            )
    values = [(item["rul_prediction"], item["anomaly_score"]) for item in predictions]
    assert values[0] != values[1], f"{dataset_id} predictions are static: {values}"
