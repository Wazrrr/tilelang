"""Compatibility exports for experiments.gemm.kernels.tiled."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("experiments.gemm.kernels.tiled")
