import pickle
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import (
    CreditApplication,
    FEATURE_MEDIANS,
    MODEL_FEATURES,
    PredictionResponse,
    TOP_SHAP_FEATURES,
)


def test_credit_application_exposes_top_shap_features_only():
    assert len(TOP_SHAP_FEATURES) == 20
    assert tuple(CreditApplication.model_fields) == TOP_SHAP_FEATURES
    assert set(TOP_SHAP_FEATURES).issubset(MODEL_FEATURES)


def test_credit_application_builds_complete_model_input_from_defaults():
    application = CreditApplication()

    model_input = application.to_model_input()

    assert tuple(model_input) == MODEL_FEATURES
    assert len(model_input) == 285
    assert model_input["AMT_CREDIT"] == application.AMT_CREDIT
    assert model_input["CNT_CHILDREN"] == FEATURE_MEDIANS["CNT_CHILDREN"]


def test_credit_application_validates_bounds_and_rejects_hidden_features():
    with pytest.raises(ValidationError):
        CreditApplication(AMT_CREDIT=1.0)

    with pytest.raises(ValidationError):
        CreditApplication(CODE_GENDER_M=2)

    with pytest.raises(ValidationError):
        CreditApplication(CNT_CHILDREN=2)


def test_schema_features_match_lightgbm_artifact():
    model_path = Path(
        "mlflow/artifacts/models/"
        "m-09098a8dc27c4199bc934fafb610fdb1/artifacts/model.pkl"
    )

    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)

    assert tuple(model.feature_name_) == MODEL_FEATURES


def test_prediction_response_validates_output_contract():
    response = PredictionResponse(
        prediction=1,
        probability_of_default=0.73,
        risk_category="High",
    )

    assert response.prediction == 1

    with pytest.raises(ValidationError):
        PredictionResponse(
            prediction=2,
            probability_of_default=0.73,
            risk_category="High",
        )

    with pytest.raises(ValidationError):
        PredictionResponse(
            prediction=1,
            probability_of_default=1.5,
            risk_category="High",
        )
