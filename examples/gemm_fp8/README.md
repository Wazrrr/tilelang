# FP8 GEMM

`example_tilelang_gemm_fp8.py` uses `T.gemm` with FP8 inputs/output and an FP32
accumulator. It computes `A @ B.T`. The compiler selects the matrix instruction
for the target; on Hopper the supported configurations use WGMMA.

## TileTune with one reference benchmark

Run from the repository root on an idle Hopper GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python -m examples.gemm_fp8.example_gemm_fp8_tiletune \
  --size 4096 --dtype float8_e4m3fn \
  --device-profile h200_device.json --output fp8_ranking.json
```

The first invocation measures fixed device primitives, independently of this
kernel's configurations, and caches them. Subsequent invocations reuse that
profile. Adding another input dtype measures only its matrix-operation probes;
memory, synchronization, scalar, exponential and reduction measurements are
shared with other kernels, including supported attention kernels.

The example analyzes all 288 configurations of the actual FP8 example, then
compiles/checks/benchmarks only the existing default configuration
(`128×128×64`, three stages, 128 consumers, rasterization disabled). That one
measurement calibrates a common latency scale and **does not change the ranking**.
If you have already measured that configuration with the same dimensions, dtype,
device and benchmark regime, pass `--reference-latency-ms <value>` to reuse it.

The report includes every configuration, pressure decisions, propagated input
tiles, pipeline/wave estimates, ties, the device profile and the latency anchor.
It applies no top-K cutoff. `--validate-all` uses the normal autotuner to benchmark
all configs. The compiler fix for a producer-barrier phase-lag hang is included;
its regression test delays an idle producer warp deliberately. Exhaustive
validation is not a calibration requirement. See the
[portable experiment instructions](../../benchmark/autotune/README.md).

Device probes support Hopper WGMMA with FP16/BF16/FP8 inputs and FP32
accumulation, and A100 MMA with FP16/BF16. A100 has no FP8 tensor-core instructions;
the portable runner records FP8 workloads as unsupported. Dtype/instruction mismatches produce unknown timing scores. The
default profile measures warm cached traffic, matching the CUDA-graph benchmark
here. See [TileTune documentation](../../docs/tiletune.md) for the streaming
profile option, API usage and limitations.
