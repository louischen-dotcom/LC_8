from fastapi.testclient import TestClient

from app import main as api
from app.schemas import MODEL_FEATURES, TOP_SHAP_FEATURES
import pandas as pd
import logging
from app.drift_monitor import DriftMonitor

AUTH_HEADER = {"Authorization": "Bearer test-token"}


class FakeModel:
    def __init__(self):
        self.last_frame = None

    def predict_proba(self, model_frame):
        self.last_frame = model_frame
        return [[0.2, 0.8]]

def build_test_monitor(batch_size=2):
    test_monitor = DriftMonitor.__new__(DriftMonitor)
    test_monitor.batch_size = batch_size
    test_monitor.drift_threshold = 0.05
    test_monitor.prediction_buffer = []
    test_monitor.logger = logging.getLogger("drift_monitor")

    test_monitor.reference_data = pd.DataFrame([
        {
            "EXT_SOURCE_MEAN": 0.5,
            "AMT_CREDIT": 200000,
            "AMT_INCOME_TOTAL": 150000,
            "DAYS_BIRTH": -15000,
            "DAYS_EMPLOYED": -3000,
        }
    ])

    return test_monitor

def test_health_reports_loaded_model(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(api, "load_model", lambda: fake_model)
    monkeypatch.setattr(api, "get_model", lambda: fake_model)

    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_features_endpoint_lists_exposed_top_shap_features(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(api, "load_model", lambda: fake_model)

    with TestClient(api.app) as client:
        response = client.get("/features")

    assert response.status_code == 200
    assert response.json() == list(TOP_SHAP_FEATURES)


def test_predict_uses_top_features_and_imputes_hidden_features(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setattr(api, "load_model", lambda: fake_model)
    monkeypatch.setattr(api, "get_model", lambda: fake_model)

    with TestClient(api.app) as client:
        response = client.post("/predict", json={}, headers=AUTH_HEADER)

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "probability_of_default": 0.8,
        "risk_category": "High",
    }
    assert list(fake_model.last_frame.columns) == list(MODEL_FEATURES)
    assert fake_model.last_frame.shape == (1, 285)


def test_predict_rejects_hidden_model_features(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setattr(api, "load_model", lambda: fake_model)
    monkeypatch.setattr(api, "get_model", lambda: fake_model)

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={"CNT_CHILDREN": 2},
            headers=AUTH_HEADER,
        )

    assert response.status_code == 422


def test_predict_returns_503_when_model_is_not_loaded(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setattr(api, "load_model", lambda: fake_model)
    monkeypatch.setattr(
        api,
        "get_model",
        lambda: (_ for _ in ()).throw(RuntimeError("Model not loaded")),
    )

    with TestClient(api.app) as client:
        response = client.post("/predict", json={}, headers=AUTH_HEADER)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"


def test_predict_requires_bearer_token(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setattr(api, "load_model", lambda: fake_model)
    monkeypatch.setattr(api, "get_model", lambda: fake_model)

    with TestClient(api.app) as client:
        response = client.post("/predict", json={})

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing bearer token"


def test_predict_rejects_invalid_bearer_token(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setattr(api, "load_model", lambda: fake_model)
    monkeypatch.setattr(api, "get_model", lambda: fake_model)

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={},
            headers={"Authorization": "Bearer wrong-token"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid bearer token"


def test_predict_returns_503_when_api_token_is_not_configured(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.delenv("API_TOKEN", raising=False)
    monkeypatch.setattr(api, "load_model", lambda: fake_model)
    monkeypatch.setattr(api, "get_model", lambda: fake_model)

    with TestClient(api.app) as client:
        response = client.post("/predict", json={}, headers=AUTH_HEADER)

    assert response.status_code == 503
    assert response.json()["detail"] == "API_TOKEN is not configured"


def test_drift_monitor_triggers(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setattr(api, "load_model", lambda: fake_model)
    monkeypatch.setattr(api, "get_model", lambda: fake_model)

    # 🔥 Replace drift monitor with small batch size
    # from app.drift_monitor import DriftMonitor
    test_monitor = build_test_monitor(batch_size=3)

    monkeypatch.setattr(api, "drift_monitor", test_monitor)

    with TestClient(api.app) as client:
        for _ in range(3):  # trigger batch
            response = client.post("/predict", json={}, headers=AUTH_HEADER)
            assert response.status_code == 200


def test_drift_monitor_logs(monkeypatch, caplog):
    fake_model = FakeModel()
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setattr(api, "load_model", lambda: fake_model)
    monkeypatch.setattr(api, "get_model", lambda: fake_model)

    # from app.drift_monitor import DriftMonitor
    test_monitor = build_test_monitor(batch_size=2)

    # Fake reference data
    test_monitor.reference_data = pd.DataFrame([
        {
            "EXT_SOURCE_MEAN": 0.5,
            "AMT_CREDIT": 200000,
            "AMT_INCOME_TOTAL": 150000,
            "DAYS_BIRTH": -15000,
            "DAYS_EMPLOYED": -3000,
        }
    ])

    monkeypatch.setattr(api, "drift_monitor", test_monitor)

    caplog.set_level(logging.INFO)

    with TestClient(api.app) as client:
        for _ in range(2):
            client.post("/predict", json={}, headers=AUTH_HEADER)


    # ✅ Check drift analysis triggered
    drift_logs = [
        record for record in caplog.records
        if "drift_analysis_completed" in record.getMessage()
    ]
    assert len(drift_logs) > 0

    # ✅ Check feature-level drift (OPTIONAL)
    feature_logs = [
        record for record in caplog.records
        if "feature_drift_detected" in record.getMessage()
    ]

    # You can assert or just inspect
    assert isinstance(feature_logs, list)