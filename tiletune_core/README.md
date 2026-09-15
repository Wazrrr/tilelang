# TileTune Core

Install independently of TileLang, TVM, PyTorch, and accelerator runtimes:

```bash
python -m pip install ./tiletune_core
```

The package has no runtime dependencies. TileLang wheels also include it.

`KernelFacts` is the versioned compiler boundary. Adapters resolve work counts,
ownership, storage, loop regions, and synchronization before exporting facts.
`BackendModel` supplies native engine service, allocation granularity, capacities,
and execution-unit residency. `AnalysisReport` records costs and structured
resource, policy, uncertainty, or unsupported diagnostics.

Native engine schedules use ordered, compressed regions. A dependency records
an operation ID and an iteration distance. An asynchronous operation releases
its issue engine before its completion latency elapses; consumers wait for its
completion. Repeated regions use max-plus exponentiation. The evaluator does
not parse expressions or infer compiler layouts.

The `ampere`, `hopper`, `blackwell`, `cdna4`, and `ascend910b` registrations define
engine/storage contracts. They **do not provide calibrated hardware profiles**.
Capacities, allocation units, and rates must be supplied explicitly. CDNA4 uses
`gfx950`; there is no CDNA3 profile fallback. Ascend exposes Cube, Vector, MTE,
L0, L1, and UB resources directly. Compiler adapters must account for resource
hierarchies (including CU/SIMD allocation) before reporting resolved peaks.

CUDA's existing numerical equations are preserved in the compatibility
evaluator `tiletune_core.cuda.evaluate_cuda_facts`. Its `cuda.v21` payload keeps
the established pipeline representation and resource estimates. The native
engine evaluator is a separate contract, not a claim that the current CUDA
compiler adapter supports native TCGEN05/TMEM or all non-CUDA schedules.

```python
import json
from tiletune_core import KernelFacts
from tiletune_core.cuda import evaluate_cuda_facts

facts = KernelFacts.from_dict(json.load(open("facts.json")))
report = evaluate_cuda_facts(facts)
print(report.score, report.units)
```

Compiler adapters must supply resolved facts using the versioned contract.
Automatic fact export and exploration depend on support in the compiler runtime.
`AttemptLedger` freezes attempts and retains failures without replacement.
