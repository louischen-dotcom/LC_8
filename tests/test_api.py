from fastapi.testclient import TestClient

from app import main as api
from app.schemas import MODEL_FEATURES, TOP_SHAP_FEATURES


AUTH_HEADER = {"Authorization": "Bearer test-token"}


class FakeModel:
    def __init__(self):
        self.last_frame = None

    def predict_proba(self, model_frame):
        self.last_frame = model_frame
        return [[0.2, 0.8]]


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
