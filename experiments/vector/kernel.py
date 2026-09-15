"""Family-owned portable implementations and independent references."""

import tilelang.language as T
from experiments._kernel import KernelCase, _random
from .reference import make_reference


def row_case(w):
    if w.op == "softmax":
        from experiments.softmax.kernel import make_case

        return make_case(w)
    (rows, columns) = (w.parameters["rows"], w.parameters["columns"])
    width = 1 << (columns - 1).bit_length()
    (dtype, op) = (w.dtype, w.op)
    epsilon = w.parameters.get("epsilon", 1e-06)

    def build(block_rows, threads, implementation="baseline", block_cols=None, vector=1, row_threads=1):
        if implementation != "baseline":
            from .schedules import row_program, elementwise_program

            if op == "elementwise" and implementation == "tiled":
                return elementwise_program(rows, columns, dtype, block_rows, block_cols, threads, vector, row_threads)
            if op != "elementwise" and implementation == "streamed":
                return row_program(rows, columns, dtype, op, epsilon, block_rows, block_cols, threads, vector, row_threads)
            raise ValueError("implementation must be baseline, streamed for reductions, or tiled for elementwise")
        if block_cols is not None or vector != 1 or row_threads != 1:
            raise ValueError("column/layout parameters require a non-baseline implementation")
        output_shape = (rows,) if op == "reduce_sum" else (rows, columns)

        @T.prim_func
        def kernel(X: T.Tensor((rows, columns), dtype), Y: T.Tensor(output_shape, dtype)):
            with T.Kernel(T.ceildiv(rows, block_rows), threads=threads) as bx:
                x = T.alloc_fragment((block_rows, width), "float32")
                values = T.alloc_fragment((block_rows, width), "float32")
                reduced = T.alloc_fragment((block_rows,), "float32")
                T.copy(X[bx * block_rows, 0], x)
                if op == "rmsnorm":
                    for i, j in T.Parallel(block_rows, width):
                        values[i, j] = x[i, j] * x[i, j]
                    T.reduce_sum(values, reduced, dim=1)
                    for i, j in T.Parallel(block_rows, width):
                        x[i, j] = x[i, j] * T.rsqrt(reduced[i] / columns + epsilon)
                elif op == "reduce_sum":
                    T.reduce_sum(x, reduced, dim=1)
                else:
                    for i, j in T.Parallel(block_rows, width):
                        x[i, j] = T.max(x[i, j] * 2 + 1, 0)
                if op == "reduce_sum":
                    T.copy(reduced, Y[bx * block_rows])
                else:
                    T.copy(x, Y[bx * block_rows, 0])

        return kernel

    return KernelCase(
        build,
        lambda device, generator: [_random((rows, columns), dtype, device, generator)],
        make_reference(w),
        [1],
        atol=1e-05 if op == "softmax" else 0.02,
    )


make_case = row_case
