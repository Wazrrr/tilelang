"""GEMM builders: the suite tiled kernel and the legacy advanced kernel."""

from .kernels.advanced import make_kernel as make_kernel, make_inputs as make_inputs
from .kernels.tiled import gemm_case as gemm_case
from .reference import reference as reference
from .spaces import advanced_configurations

make_case = gemm_case


def get_configs():
    return advanced_configurations()
