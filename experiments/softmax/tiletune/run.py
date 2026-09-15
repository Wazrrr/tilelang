"""Run the softmax study with the shared comparison protocol."""

from experiments.common.family_cli import comparison_main


def main(argv=None):
    return comparison_main("softmax", argv)


if __name__ == "__main__":
    raise SystemExit(main())
