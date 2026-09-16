"""Audit the declared grouped GEMM configuration space in isolated shards."""

from experiments.common.family_cli import census_main


def main(argv=None):
    return census_main("grouped_gemm", argv)


if __name__ == "__main__":
    raise SystemExit(main())
