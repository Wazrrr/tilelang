"""Audit the declared flash_attention configuration space in isolated shards."""

from experiments.common.family_cli import census_main


def main(argv=None):
    return census_main("flash_attention", argv)


if __name__ == "__main__":
    raise SystemExit(main())
