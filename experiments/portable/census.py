"""Compatibility entry point for experiments.common.census."""

import sys
from importlib import import_module

_impl = import_module("experiments.common.census")

if __name__ == "__main__":
    raise SystemExit(_impl.main())
else:
    sys.modules[__name__] = _impl
