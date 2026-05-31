import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app.model_loader as model_loader
from app.main import build_model_array, predict_default_probability
from app.model_loader import load_model
from app.schemas import CreditApplication


def benchmark(fn, runs=1000, warmup=50):
    for _ in range(warmup):
        fn()

    times_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - start) * 1000)

    return {
        "mean": statistics.mean(times_ms),
        "median": statistics.median(times_ms),
        "min": min(times_ms),
        "max": max(times_ms),
    }


def main():
    os.environ["MODEL_RUNTIME"] = "lightgbm"
    model_loader._model = None
    model = load_model()
    application = CreditApplication()
    model_array = build_model_array(application).astype(np.float32)

    onnx_path = PROJECT_ROOT / "models" / "onnx" / "home_credit_lightgbm.onnx"
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    def predict_lightgbm_numpy():
        return predict_default_probability(model, model_array)

    def predict_onnx():
        outputs = session.run(output_names, {input_name: model_array})
        prediction = int(outputs[0][0])
        probability = float(outputs[1][0][1])
        return prediction, probability

    lgb_prediction, lgb_probability = predict_lightgbm_numpy()
    onnx_prediction, onnx_probability = predict_onnx()

    print("Prediction equivalence")
    print(f"LightGBM prediction: {lgb_prediction}, probability: {lgb_probability:.10f}")
    print(f"ONNX prediction:     {onnx_prediction}, probability: {onnx_probability:.10f}")
    print(f"Probability diff:    {abs(lgb_probability - onnx_probability):.12f}")

    lgb_stats = benchmark(predict_lightgbm_numpy)
    onnx_stats = benchmark(predict_onnx)

    print("")
    print("LightGBM NumPy benchmark")
    for key, value in lgb_stats.items():
        print(f"{key}: {value:.4f} ms")

    print("")
    print("ONNX Runtime benchmark")
    for key, value in onnx_stats.items():
        print(f"{key}: {value:.4f} ms")

    improvement = (lgb_stats["mean"] - onnx_stats["mean"]) / lgb_stats["mean"] * 100
    print("")
    print(f"ONNX mean latency improvement vs LightGBM NumPy: {improvement:.2f}%")


if __name__ == "__main__":
    main()