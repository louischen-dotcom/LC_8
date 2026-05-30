import statistics
import sys
import time
from pathlib import Path
import warnings

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import build_model_frame, predict_default_probability
from app.model_loader import load_model
from app.schemas import CreditApplication, MODEL_FEATURES


def benchmark(fn, runs=1000, warmup=20):
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
    model = load_model()
    application = CreditApplication()
    model_frame = build_model_frame(application)

    model_input = application.to_model_input()
    model_array = np.array(
        [[model_input[feature] for feature in MODEL_FEATURES]],
        dtype=np.float32,
    )

    def predict_dataframe():
        return predict_default_probability(model, model_frame)

    def predict_numpy():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            probabilities = model.predict_proba(model_array)

        probability = float(probabilities[0][1])
        prediction = int(probability >= 0.5)
        return prediction, probability

    df_prediction, df_probability = predict_dataframe()
    np_prediction, np_probability = predict_numpy()

    print("Prediction equivalence")
    print(f"DataFrame prediction: {df_prediction}, probability: {df_probability:.8f}")
    print(f"NumPy prediction:     {np_prediction}, probability: {np_probability:.8f}")
    print(f"Probability diff:     {abs(df_probability - np_probability):.10f}")

    df_stats = benchmark(predict_dataframe)
    np_stats = benchmark(predict_numpy)

    print("")
    print("DataFrame benchmark")
    for key, value in df_stats.items():
        print(f"{key}: {value:.4f} ms")

    print("")
    print("NumPy benchmark")
    for key, value in np_stats.items():
        print(f"{key}: {value:.4f} ms")

    improvement = (
        (df_stats["mean"] - np_stats["mean"]) / df_stats["mean"] * 100
    )
    print("")
    print(f"Mean latency improvement: {improvement:.2f}%")


if __name__ == "__main__":
    main()