"""Compatibility entry point for experiments.common.audit_model."""

import sys
from importlib import import_module

if not __package__:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_impl = import_module("experiments.common.audit_model")

if __name__ == "__main__":
    raise SystemExit(_impl.main())
else:
    sys.modules[__name__] = _impl
