# Experiment utilities

Reusable helpers live here; coordinators and worker entry points live in
`experiments/common/`, and kernel definitions/pools remain family-owned.

| Module | Responsibility |
| --- | --- |
| `cli.py` | Config selection, device metadata, active-source fingerprints |
| `kernel.py` | KernelCase contract, input helpers and numerical checks |
| `tiletune.py` | Saved TileTune ranking and winner helpers |
| `monitor.py`, `locking.py` | GPU observations, contention rejection and device leases |
| `baseline_store.py` | Family/GPU storage, explicit collection/refresh, and read-only verified baseline reuse |
| `results.py`, `io.py` | Backend-independent result reading, Oracle@K and atomic JSON writing |
| `diagnostics.py` | Ranking coverage and oracle diagnostics |
| `grid.py`, `subsets.py` | Grid enumeration and deterministic development subsets |

Import these modules directly. The old root-level helper modules and moved
`common/` aliases are removed. Recorded source snapshots remain historical data.
