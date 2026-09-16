"""Family entry points for the shared suite and census protocols."""


def comparison_main(family, argv=None):
    from experiments.suite import main

    return main(argv, family=family)


def census_main(family, argv=None):
    from .census import main

    return main(argv, family=family)
