"""Compatibility entry point for experiments.common.subsets."""

import sys
from importlib import import_module

_impl = import_module("experiments.common.subsets")
sys.modules[__name__] = _impl
