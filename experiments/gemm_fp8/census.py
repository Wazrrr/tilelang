"""Audit the complete FP8 GEMM configuration pool."""

from experiments.common.family_cli import census_main


def main(argv=None):
    return census_main("gemm_fp8", argv)


if __name__ == "__main__":
    raise SystemExit(main())
