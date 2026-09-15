"""Compatibility entry point for experiments.common.kernels."""

import sys
from importlib import import_module

_impl = import_module("experiments.common.kernels")
sys.modules[__name__] = _impl
