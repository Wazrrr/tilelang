"""Compatibility entry point for experiments.common.diagnostics."""

import sys
from importlib import import_module

_impl = import_module("experiments.common.diagnostics")
sys.modules[__name__] = _impl
