# Compatibility entry points

Start with the [experiment overview](../README.md) or a kernel family:
[GEMM](../gemm/README.md), [attention](../flash_attention/README.md),
[KDA](../kda/README.md), [softmax](../softmax/README.md).

The implementation now lives under `experiments/common/` and the family folders.
Existing Python imports and `python -m experiments.portable.<command>` commands
forward to the same implementations. The old manifest paths link to the
canonical files in `experiments/manifests/`.

| Previous entry | Canonical entry |
| --- | --- |
| `experiments.portable.suite` | `experiments.suite` |
| `experiments.portable.compare` | `experiments.common.comparison` |
| `experiments.portable.run` | `experiments.common.run` |
| `experiments.portable.census` | `experiments.common.census` |
| `experiments.portable.gemm_service_audit` | `experiments.gemm.service_audit` |

Other shared modules retain their names under `experiments.common`.
The [shared runner reference](../common/README.md) documents custom manifests,
workers, standalone comparisons, diagnostics, and historical studies.
