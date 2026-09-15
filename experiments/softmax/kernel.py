"""Softmax inputs and dispatch between full-row and streamed implementations."""

from experiments._kernel import KernelCase, _random
from .reference import reference


def make_case(w):
    rows, columns, dtype = w.parameters["rows"], w.parameters["columns"], w.dtype

    def build(block_rows, threads, implementation="baseline", block_cols=None, vector=1, row_threads=1):
        if implementation == "streamed":
            from .kernels.streamed import softmax_program

            return softmax_program(rows, columns, dtype, block_rows, block_cols, threads, vector, row_threads)
        if implementation != "baseline":
            raise ValueError("softmax implementation must be baseline or streamed")
        if block_cols is not None or vector != 1 or row_threads != 1:
            raise ValueError("column/layout parameters require a non-baseline implementation")
        from .kernels.baseline import softmax_program

        return softmax_program(rows, columns, dtype, block_rows, threads)

    return KernelCase(build, lambda device, generator: [_random((rows, columns), dtype, device, generator)], reference, [1], atol=1e-5)
