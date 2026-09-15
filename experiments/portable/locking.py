"""Compatibility entry point for experiments.common.locking."""

import sys
from importlib import import_module

_impl = import_module("experiments.common.locking")
sys.modules[__name__] = _impl
