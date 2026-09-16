"""Compare tuning methods on the single expanded GEMM pool."""

from experiments.common.family_cli import comparison_main


def main(argv=None):
    return comparison_main("gemm", argv)


if __name__ == "__main__":
    raise SystemExit(main())
