# app/model_loader.py

import mlflow.pyfunc
import os
from pathlib import Path

_model = None

def load_model():
    """Load the model ONCE at startup from MLflow Registry."""
    global _model

    # Configuration via environment variables
    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{Path.cwd() / 'mlflow' / 'mlflow.db'}"
    )
    model_name = os.environ.get("MODEL_NAME", "home_credit_scoring")
    model_version = os.environ.get("MODEL_VERSION", "1")

    try:
        # Configure MLflow tracking
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Load model from MLflow Registry
        model_uri = f"models:/{model_name}/{model_version}"
        _model = mlflow.pyfunc.load_model(model_uri)
        
        print(f"✓ Model loaded from MLflow: {model_uri}")
        print(f"  Tracking URI: {mlflow_tracking_uri}")
        return _model
    
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{model_name}' v{model_version} from {mlflow_tracking_uri}. "
            f"Error: {e}"
        )
    

def get_model():
    """Get the loaded model."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model