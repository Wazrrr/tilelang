"""Compatibility entry point for experiments.common.acceptance."""

import sys
from importlib import import_module

_impl = import_module("experiments.common.acceptance")
sys.modules[__name__] = _impl
