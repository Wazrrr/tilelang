"""Compatibility exports for experiments.flash_attention.kernel."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("experiments.flash_attention.kernel")
