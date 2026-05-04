import pandas as pd

from app.model_loader import load_model
from app.schemas import CreditApplication, MODEL_FEATURES


def build_default_model_frame() -> pd.DataFrame:
    application = CreditApplication()
    return pd.DataFrame([application.to_model_input()], columns=MODEL_FEATURES)


def test_model_loads():
    model = load_model()

    assert model is not None


def test_model_has_predict_methods():
    model = load_model()

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_returns_one_prediction():
    model = load_model()
    test_input = build_default_model_frame()

    prediction = model.predict(test_input)

    assert len(prediction) == 1
    assert int(prediction[0]) in {0, 1}


def test_model_returns_one_default_probability():
    model = load_model()
    test_input = build_default_model_frame()

    probabilities = model.predict_proba(test_input)
    probability_of_default = float(probabilities[0][1])

    assert probabilities.shape == (1, 2)
    assert 0.0 <= probability_of_default <= 1.0
