import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm


_model: Any | None = None


def _default_local_model_path() -> Path:
    artifact_root = Path.cwd() / "mlflow" / "artifacts" / "models"
    candidates = sorted(artifact_root.glob("*/artifacts/MLmodel"))
    if not candidates:
        raise FileNotFoundError(f"No local MLflow model found under {artifact_root}")
    return candidates[0].parent


def load_model() -> Any:
    """Load the model once, using LOCAL_MODEL_URI or MLflow Registry fallback."""
    global _model

    if _model is not None:
        return _model

    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{Path.cwd() / 'mlflow' / 'mlflow.db'}",
    )
    model_name = os.environ.get("MODEL_NAME", "home_credit_scoring")
    model_version = os.environ.get("MODEL_VERSION", "1")
    local_model_uri = os.environ.get("LOCAL_MODEL_URI")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    registry_model_uri = f"models:/{model_name}/{model_version}"

    if local_model_uri:
        try:
            _model = mlflow.lightgbm.load_model(local_model_uri)
            print(f"Model loaded from local MLflow artifact: {local_model_uri}")
            return _model
        except Exception as local_error:
            raise RuntimeError(
                f"Failed to load model from LOCAL_MODEL_URI '{local_model_uri}'. "
                f"Error: {local_error}"
            ) from local_error

    try:
        _model = mlflow.lightgbm.load_model(registry_model_uri)
        print(f"Model loaded from MLflow Registry: {registry_model_uri}")
        return _model
    except Exception as registry_error:
        fallback_path = Path(local_model_uri) if local_model_uri else _default_local_model_path()
        try:
            _model = mlflow.lightgbm.load_model(str(fallback_path))
            print(f"Model loaded from local MLflow artifact: {fallback_path}")
            return _model
        except Exception as local_error:
            raise RuntimeError(
                f"Failed to load model from registry '{registry_model_uri}' "
                f"or local artifact '{fallback_path}'. "
                f"Registry error: {registry_error}. Local error: {local_error}"
            ) from local_error


def get_model() -> Any:
    """Get the loaded model."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model
