"""Run the kda study with the shared comparison protocol."""

from experiments.common.family_cli import comparison_main


def main(argv=None):
    return comparison_main("kda", argv)


if __name__ == "__main__":
    raise SystemExit(main())
