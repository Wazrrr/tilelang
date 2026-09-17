"""System ablations for the final FP8 GEMM example cases."""

from experiments.common.system import main


if __name__ == "__main__":
    raise SystemExit(main(family="gemm_fp8"))
