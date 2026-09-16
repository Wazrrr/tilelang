"""Compare tuning methods on the single expanded grouped GEMM pool."""

from experiments.common.family_cli import comparison_main


def main(argv=None):
    return comparison_main("grouped_gemm", argv)


if __name__ == "__main__":
    raise SystemExit(main())
