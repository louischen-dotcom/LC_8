# Performance Optimization Report

## Context

The credit scoring API was already deployed with structured prediction logging,
drift monitoring, and operational latency tracking. The optimization objective
was to reduce inference latency without changing model outputs or the API
contract.

Runtime before optimization:

- API: FastAPI
- Model: MLflow LightGBM model `models:/home_credit_scoring/1`
- Monitoring: prediction logging, drift monitoring, Evidently 0.7.21
- Hardware for local tests: developer workstation CPU

## Baseline Measurement

The initial local inference benchmark measured the model hot path with the
existing pandas `DataFrame` input.

```text
Runs: 100
Mean latency: 1.9436 ms
Median latency: 1.8421 ms
Min latency: 1.4010 ms
Max latency: 3.6444 ms
```

The benchmark excluded network overhead and focused on local model inference.

## Profiling Results

`cProfile` was used to profile 1000 repeated predictions after model loading.
This step was run before testing ONNX to avoid optimizing blindly. At that
stage, the production path used the LightGBM Python runtime with pandas input,
so the goal was to identify whether latency came from the model computation,
input preparation, API overhead, logging, or monitoring.

Top cumulative-time entries:

```text
1000 calls main.py:55(predict_default_probability)  3.648 s
1000 calls sklearn.py:1615(predict_proba)           3.643 s
1000 calls sklearn.py:1093(predict)                 3.599 s
1000 calls basic.py:4701(predict)                   2.572 s
1000 calls basic.py:827(_data_from_pandas)          1.996 s
1000 calls basic.py:810(_pandas_to_numpy)           1.328 s
1000 calls basic.py:798(_check_for_bad_pandas_dtypes) 1.204 s
```

The main bottleneck was not the tree inference itself, but the pandas-to-LightGBM
input conversion and dtype validation.

The initial `cProfile` results did not show ONNX-related gains because ONNX
Runtime was not yet part of the inference path. `cProfile` is also most useful
for Python-level function calls; native runtimes such as ONNX Runtime execute
most work inside optimized C/C++ code and usually appear as a compact
`session.run(...)` call from Python. For that reason, profiling was used to
understand the existing LightGBM path, while benchmarks were used to compare
runtime alternatives.

## Optimization Tested

The optimized version builds a NumPy array directly from the validated
`CreditApplication`, using the exact `MODEL_FEATURES` order expected by the
model.

Before:

```text
CreditApplication -> pandas DataFrame -> LightGBM predict_proba
```

After:

```text
CreditApplication -> NumPy array -> LightGBM predict_proba
```

The API still validates inputs with Pydantic and still logs predictions and drift
monitoring records. Only the internal model input representation changed.

## Accuracy And Regression Check

The DataFrame and NumPy paths were compared on multiple valid API inputs.

```text
Case 1 diff: 0.000000000000
Case 2 diff: 0.000000000000
Case 3 diff: 0.000000000000
Case 4 diff: 0.000000000000
Case 5 diff: 0.000000000000

All cases matched. Max probability diff: 0.000000000000
```

Predicted classes and probabilities were identical for the tested cases.

## NumPy Benchmark

The final comparison was run in the same Python process to reduce noise between
runs.

```text
DataFrame benchmark
mean: 1.5190 ms
median: 1.4541 ms
min: 1.3538 ms
max: 2.4473 ms

NumPy benchmark
mean: 1.0146 ms
median: 0.9700 ms
min: 0.8932 ms
max: 2.1252 ms

Mean latency improvement: 33.21%
```

The optimized hot-path benchmark with 1000 measured runs produced:

```text
Runs: 1000
Mean latency: 0.9596 ms
Median latency: 0.9230 ms
Min latency: 0.8809 ms
Max latency: 1.6724 ms
```

## ONNX Runtime Evaluation

The LightGBM model was exported to ONNX with `onnxmltools` and
`target_opset=15`. Opset 15 was selected because the installed LightGBM
converter supports ONNX opsets up to 15 for this model type.

ONNX Runtime was then benchmarked against the optimized LightGBM NumPy path in
the same Python process.

```text
LightGBM NumPy benchmark
mean: 0.9951 ms
median: 0.9348 ms
min: 0.8677 ms
max: 2.1814 ms

ONNX Runtime benchmark
mean: 0.0305 ms
median: 0.0271 ms
min: 0.0238 ms
max: 0.9632 ms

ONNX mean latency improvement vs LightGBM NumPy: 96.94%
```

Prediction equivalence was checked on five valid API inputs.

```text
Max probability diff: 0.000000131269
Tolerance: 1e-05
Predicted classes: identical
```

The small probability differences are expected numerical differences between
the LightGBM Python runtime and ONNX Runtime. They remained far below the
validation tolerance and did not change predicted classes.

The ONNX benchmark measures the model hot path only. The full `/predict` HTTP
latency also includes FastAPI routing, Pydantic validation, authentication,
JSON serialization, logging, drift monitoring, and network overhead. Therefore,
the full API response time is expected to improve less than the pure model
runtime, even though the model inference step itself is much faster.

## Monitoring Notes

Evidently was enabled with the modern 0.7 API:

```python
from evidently import Report
from evidently.presets import DataDriftPreset
```

A local drift check confirmed:

```text
evidently_available: True
batch_size: 30
```

Small synthetic batches may produce NumPy statistical warnings during drift
tests. Production drift monitoring should use sufficiently large batches.

## Validation

Automated tests after ONNX integration:

```text
23 passed, 18 warnings
```

The warnings are emitted by Evidently/NumPy drift calculations on small test
batches and are not related to the optimized inference path.

## Final Configuration

The final configuration uses ONNX Runtime as the default inference runtime and
keeps LightGBM available as a rollback/fallback option.

Runtime selection is controlled by:

```text
MODEL_RUNTIME=onnx
MODEL_RUNTIME=lightgbm
```

The ONNX model path can be overridden with:

```text
ONNX_MODEL_PATH
```

ONNX Runtime was selected because:

- profiling first identified pandas conversion as the initial bottleneck;
- replacing pandas with NumPy reduced LightGBM inference latency by about 33%;
- ONNX Runtime then reduced the optimized model hot-path latency by about 97%;
- prediction classes stayed identical across the validation cases;
- maximum observed probability difference was only `0.000000131269`;
- LightGBM remains available through `MODEL_RUNTIME=lightgbm` if rollback is
  needed.

This configuration improves response-time headroom while preserving the existing
API contract, monitoring pipeline, and CI/CD deployment flow.
