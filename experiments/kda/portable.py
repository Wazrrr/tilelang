"""Compatibility exports for experiments.kda.kernel."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("experiments.kda.kernel")
