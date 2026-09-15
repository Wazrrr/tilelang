"""Compatibility entry point for experiments.common.repair_study."""

import sys
from importlib import import_module

_impl = import_module("experiments.common.repair_study")

if __name__ == "__main__":
    raise SystemExit(_impl.main())
else:
    sys.modules[__name__] = _impl
