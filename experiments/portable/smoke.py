"""Compatibility entry point for experiments.common.smoke."""

import sys
from importlib import import_module

_impl = import_module("experiments.common.smoke")
sys.modules[__name__] = _impl
