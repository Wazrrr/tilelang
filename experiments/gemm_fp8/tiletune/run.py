"""Compare tuning methods on the FP8 GEMM example pool."""

from experiments.common.family_cli import comparison_main


def main(argv=None):
    return comparison_main("gemm_fp8", argv)


if __name__ == "__main__":
    raise SystemExit(main())
