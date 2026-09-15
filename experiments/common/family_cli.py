"""Small entry points sharing the suite and census protocols."""

from importlib import import_module
import sys

# Existing GEMM/attention commands keep their fixed-grid behavior. Named-suite
# options explicitly select the family study; --help documents the new interface.
STUDY_OPTIONS = {
    "--suite",
    "--device",
    "--devices",
    "--device-manifest",
    "--config-space",
    "--plan",
    "--freeze",
    "--development-report",
    "--help",
    "-h",
}


def comparison_main(family, argv=None, *, legacy_module=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if legacy_module and ("--legacy" in argv or not any(arg.split("=", 1)[0] in STUDY_OPTIONS for arg in argv)):
        argv = [arg for arg in argv if arg != "--legacy"]
        return import_module(legacy_module).legacy_main(argv)
    from experiments.suite import main

    return main(argv, family=family)


def census_main(family, argv=None):
    from .census import main

    return main(argv, family=family)
