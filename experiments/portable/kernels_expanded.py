"""Compatibility entry point for experiments.common.kernels_expanded."""

import sys
from importlib import import_module

_impl = import_module("experiments.common.kernels_expanded")
sys.modules[__name__] = _impl
