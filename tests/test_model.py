import pandas as pd
import pytest

import app.model_loader as model_loader
from app.model_loader import load_model
from app.schemas import CreditApplication, MODEL_FEATURES


@pytest.fixture(autouse=True)
def reset_loaded_model():
    model_loader._model = None
    yield
    model_loader._model = None


def build_default_model_frame() -> pd.DataFrame:
    application = CreditApplication()
    return pd.DataFrame([application.to_model_input()], columns=MODEL_FEATURES)


def test_model_loads():
    model = load_model()

    assert model is not None


def test_lightgbm_model_has_predict_methods(monkeypatch):
    monkeypatch.setenv("MODEL_RUNTIME", "lightgbm")
    model = load_model()

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_lightgbm_model_returns_one_prediction(monkeypatch):
    monkeypatch.setenv("MODEL_RUNTIME", "lightgbm")
    model = load_model()
    test_input = build_default_model_frame()

    prediction = model.predict(test_input)

    assert len(prediction) == 1
    assert int(prediction[0]) in {0, 1}


def test_lightgbm_model_returns_one_default_probability(monkeypatch):
    monkeypatch.setenv("MODEL_RUNTIME", "lightgbm")
    model = load_model()
    test_input = build_default_model_frame()

    probabilities = model.predict_proba(test_input)
    probability_of_default = float(probabilities[0][1])

    assert probabilities.shape == (1, 2)
    assert 0.0 <= probability_of_default <= 1.0


def test_onnx_model_loads(monkeypatch):
    monkeypatch.setenv("MODEL_RUNTIME", "onnx")
    model = load_model()

    assert hasattr(model, "run")
    assert len(model.get_inputs()) == 1
    assert len(model.get_outputs()) == 2