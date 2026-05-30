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

## Final Benchmark

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

Automated tests after optimization:

```text
22 passed, 18 warnings
```

The warnings are emitted by Evidently/NumPy drift calculations on small test
batches and are not related to the optimized inference path.

## Final Configuration Justification

The final configuration keeps LightGBM as the inference runtime and uses NumPy
arrays for model input.

This was selected instead of ONNX because:

- the LightGBM model was already very fast;
- profiling identified pandas conversion as the bottleneck;
- NumPy input reduced mean inference latency by about 33%;
- the change is small and compatible with the existing MLflow model;
- no model conversion or additional production runtime dependency is required;
- prediction equivalence was verified on valid API inputs.

ONNX Runtime remains a possible future optimization if larger models or higher
traffic make the current runtime insufficient.
