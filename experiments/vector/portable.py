"""Compatibility exports for experiments.vector.kernel."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("experiments.vector.kernel")
