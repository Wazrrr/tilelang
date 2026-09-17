"""Compare methods on the same FP8 example and configuration pool."""

from experiments.common.family_cli import comparison_main


def main(argv=None):
    return comparison_main("gemm_fp8", argv)


if __name__ == "__main__":
    raise SystemExit(main())
