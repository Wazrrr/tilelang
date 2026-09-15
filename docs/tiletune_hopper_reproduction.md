# Reproduce the TileTune study on Hopper

Run these commands from the repository root on the Hopper machine, using the
committed experiment configuration and TileTune adapter refactor together.
The full suite uses eight held-out cases across GEMM, attention, chunk-output
KDA, and softmax; eight training cases; four validation cases; and seeds 123,
456, and 789. The `large` preset caps each generated pool at 1,024 configurations
while retaining the smaller 691-configuration softmax pool.

## Environment and device

Activate the intended Python environment and configure `CUDA_HOME` and `CXX`
for that machine's CUDA toolkit and supported C++ compiler. Install from this
checkout, including the XGBoost dependency used by the comparison:

```bash
python -m pip install . xgboost
```

Use an explicit device manifest for H100 or H200. The built-in `--devices hopper`
entry requires an H200; this manifest accepts either device and still verifies
the Hopper architecture:

```bash
cat > /tmp/tiletune-hopper-device.json <<'JSON'
[
  {
    "name": "hopper",
    "target": {"kind": "cuda", "arch": "sm_90a"},
    "expected_device_pattern": "H100|H200"
  }
]
JSON
```

## Plan and run

Inspect the complete plan without accessing the GPU:

```bash
python -m experiments.suite --suite full \
  --device-manifest /tmp/tiletune-hopper-device.json --plan
```

Check correctness and the worker setup with the small smoke suite:

```bash
python -m experiments.suite --suite smoke \
  --device-manifest /tmp/tiletune-hopper-device.json \
  --workers 16 --output experiments/results/hopper-smoke
```

Collect the full comparison:

```bash
python -m experiments.suite --suite full \
  --device-manifest /tmp/tiletune-hopper-device.json \
  --workers 16 --warmup 10 --rep 50 --timeout 60 --case-timeout 86400 \
  --output experiments/results/hopper-full
```

`full` runs the final shapes and three-seed protocol without requiring an earlier
development acceptance report. It measures Hopper primitive profiles locally,
collects the seeded XGBoost samples, freezes TileTune/random/XGBoost selections
for every seed, and then collects the shared exhaustive references and seven
winner remeasurements. `pipeline_time` is the TileTune ranking metric. Profiles
and measured references must match the actual device and experiment identity.

The top-level suite resumes completed work when the exact command is repeated.
Its frozen source, profile, and setting checks require a new output directory
after changing the code or protocol. Save `study-lock.json`, `acceptance.json`,
and the per-case `comparison.json` and worker artifacts with the results.

The `exhaustive` preset preserves the older uncapped domains for common
run/comparison/census commands. Named `full` and `final` suites use the capped
`large` pools, so their results have a different candidate-pool identity from
historical uncapped studies. H100 and H200 measurements also have separate device
identities. Hopper execution and performance must be verified on the target
machine; local A100 and offline checks do not establish Hopper performance.
