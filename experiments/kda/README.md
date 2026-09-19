# KDA chunk output

The KDA experiment family contains only `kda_chunk_o`, using the repository's
`examples/kda/chunk_o.py` builder on A100, H200 and B200.

Q, V, attention, per-chunk state and output are BF16; accumulation is FP32.
Named workloads fix DK=DV=128 and chunk size=64. FP32 base-2 cumulative log gates
reset at each chunk boundary, and query scale is 128^-0.5. The state input has
shape (B,S/64,H,128,128). Gate generation is outside the timed kernel.

The shared ordered pool has **1,296 configurations**:

| Axis | Choices |
| --- | --- |
| Key tile | 16, 32, 48, 64, 96, 128 |
| Value tile | 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256 |
| Pipeline stages | 0, 1, 2, 3, 4, 5 |
| Threads | 64, 128, 256 |

`cases.py` owns shapes; `kernel.py`, `spaces.py`, `reference.py` and `carver.py`
own inputs/builder adaptation, the pool, the independent Torch reference and
Carver integration. Candidates that fail compilation or correctness remain
recorded failures; the pool is not pruned per architecture.

The five final (B,H,S) shapes are (1,32,2048), (1,64,4096), (1,32,8192),
(2,32,4096) and (1,64,16384). Two training shapes and one validation shape are
disjoint from these final holdouts.

```bash
python -m experiments.kda.tiletune.run --suite full --device hopper --plan
python -m experiments.kda.system.run --plan
```

Only chunk output is timed. Intra, inter-solve, WY and recurrent state updates
are outside the active suite; the original standalone examples remain available.
See [the common contract](../BENCHMARK_CONTRACT.md) for all family shapes and
permitted backend differences.
