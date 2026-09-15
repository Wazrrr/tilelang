"""Compatibility entry point for experiments.common.execution."""

import sys
from importlib import import_module

_impl = import_module("experiments.common.execution")
sys.modules[__name__] = _impl
