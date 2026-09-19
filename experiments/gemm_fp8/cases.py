"""The same aligned dense shapes as BF16 GEMM, with E4M3 storage."""

from dataclasses import replace


def _fp8(workloads):
    return [replace(w, name=w.name.replace("gemm_", "gemm_fp8_", 1), op="gemm_fp8", dtype="float8_e4m3fn") for w in workloads]


def cases(holdout=False):
    from experiments.gemm.cases import cases as dense_cases

    return _fp8(dense_cases(holdout))


def training_cases():
    from experiments.gemm.cases import training_cases as dense_training_cases

    return _fp8(dense_training_cases())
