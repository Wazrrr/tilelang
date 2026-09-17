"""System ablations for FP8 GEMM."""

from experiments.common.system import main


if __name__ == "__main__":
    raise SystemExit(main(family="gemm_fp8"))
