"""Compatibility entry point for experiments.gemm.service_audit."""

import sys
from importlib import import_module

_impl = import_module("experiments.gemm.service_audit")

if __name__ == "__main__":
    raise SystemExit(_impl.main())
else:
    sys.modules[__name__] = _impl
