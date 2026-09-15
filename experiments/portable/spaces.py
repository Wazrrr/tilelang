"""Compatibility entry point for experiments.common.spaces."""

import sys
from importlib import import_module

_impl = import_module("experiments.common.spaces")
sys.modules[__name__] = _impl
