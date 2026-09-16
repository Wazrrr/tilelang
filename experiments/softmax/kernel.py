"""Use the online-softmax example directly, including its log2/exp2 recurrence."""

from threading import Lock
import tilelang.language as T
from experiments.utils.kernel import KernelCase, _random
from .reference import reference

_SOFTMAX_LOCK = Lock()


def _softmax_program(rows, columns, dtype, block_rows, block_cols, threads):
    from examples.online_softmax.online_softmax import softmax_kernel

    with _SOFTMAX_LOCK:
        return softmax_kernel.get_tir(
            T.Tensor((rows, columns), dtype), BLOCK_M=block_rows, BLOCK_N=block_cols, dtype=dtype, threads=threads
        )


def make_case(w):
    rows, columns, dtype = w.parameters["rows"], w.parameters["columns"], w.dtype

    def build(BLOCK_M, BLOCK_N, threads):
        return _softmax_program(rows, columns, dtype, BLOCK_M, BLOCK_N, threads)

    # Eager T.empty already records the output index in the example's PrimFunc.
    return KernelCase(build, lambda device, generator: [_random((rows, columns), dtype, device, generator)], reference, None, atol=1e-5)
