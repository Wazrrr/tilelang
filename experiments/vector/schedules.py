"""Additional scheduling choices owned by this kernel family."""

import tilelang.language as T
from experiments._kernel import _row_layout


def row_program(rows, columns, dtype, op, epsilon, block_rows, block_cols, threads, vector, row_threads):
    if op == "softmax":
        from experiments.softmax.kernels.streamed import softmax_program

        return softmax_program(rows, columns, dtype, block_rows, block_cols, threads, vector, row_threads)
    (matrix_layout, reduced_layout) = _row_layout(block_rows, block_cols, threads, vector, row_threads)
    output_shape = (rows,) if op == "reduce_sum" else (rows, columns)

    @T.prim_func
    def kernel(X: T.Tensor((rows, columns), dtype), Y: T.Tensor(output_shape, dtype)):
        with T.Kernel(T.ceildiv(rows, block_rows), threads=threads) as bx:
            x = T.alloc_fragment((block_rows, block_cols), "float32")
            values = T.alloc_fragment((block_rows, block_cols), "float32")
            partial = T.alloc_fragment((block_rows,), "float32")
            total = T.alloc_fragment((block_rows,), "float32")
            maximum = T.alloc_fragment((block_rows,), "float32")
            next_maximum = T.alloc_fragment((block_rows,), "float32")
            T.annotate_layout(
                {
                    x: matrix_layout,
                    values: matrix_layout,
                    partial: reduced_layout,
                    total: reduced_layout,
                    maximum: reduced_layout,
                    next_maximum: reduced_layout,
                }
            )
            T.clear(total)
            T.fill(maximum, -T.infinity("float32"))
            for tile in T.serial(T.ceildiv(columns, block_cols)):
                T.copy(X[bx * block_rows, tile * block_cols], x, coalesced_width=vector)
                for i, j in T.Parallel(block_rows, block_cols):
                    if op == "rmsnorm":
                        values[i, j] = x[i, j] * x[i, j]
                    else:
                        values[i, j] = x[i, j]
                T.reduce_sum(values, partial, dim=1)
                for i in T.Parallel(block_rows):
                    total[i] += partial[i]
            if op == "reduce_sum":
                T.copy(total, Y[bx * block_rows])
            else:
                for tile in T.serial(T.ceildiv(columns, block_cols)):
                    T.copy(X[bx * block_rows, tile * block_cols], x, coalesced_width=vector)
                    for i, j in T.Parallel(block_rows, block_cols):
                        values[i, j] = x[i, j] * T.rsqrt(total[i] / columns + epsilon)
                    T.copy(values, Y[bx * block_rows, tile * block_cols], coalesced_width=vector)

    return kernel


def elementwise_program(rows, columns, dtype, block_rows, block_cols, threads, vector, row_threads):
    layout, _ = _row_layout(block_rows, block_cols, threads, vector, row_threads)

    @T.prim_func
    def kernel(X: T.Tensor((rows, columns), dtype), Y: T.Tensor((rows, columns), dtype)):
        with T.Kernel(T.ceildiv(columns, block_cols), T.ceildiv(rows, block_rows), threads=threads) as (bx, by):
            x = T.alloc_fragment((block_rows, block_cols), "float32")
            T.annotate_layout({x: layout})
            T.copy(X[by * block_rows, bx * block_cols], x, coalesced_width=vector)
            for i, j in T.Parallel(block_rows, block_cols):
                x[i, j] = T.max(x[i, j] * 2 + 1, 0)
            T.copy(x, Y[by * block_rows, bx * block_cols], coalesced_width=vector)

    return kernel
