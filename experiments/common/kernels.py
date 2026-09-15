"""Build a kernel through its owning family."""

from experiments.families import family_module
from experiments._kernel import KernelCase as KernelCase, _random as _random
from experiments.gemm.kernel import gemm_case as gemm_case
from experiments.flash_attention.kernel import attention_case as attention_case, _attention_program as _attention_program
from experiments.kda.kernel import kda_recurrent_case as kda_recurrent_case, kda_chunk_case as kda_chunk_case
from experiments.kda.kernels.tiled import _kda_recurrent_tiled as _kda_recurrent_tiled, _kda_chunk_tiled as _kda_chunk_tiled
from experiments.vector.kernel import row_case as row_case


def make_case(workload):
    return family_module(workload.op, "kernel").make_case(workload)
