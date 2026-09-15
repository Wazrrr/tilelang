"""Column-tiled softmax with full-row normalization."""

import tilelang.language as T
from experiments._kernel import _row_layout


def softmax_program(rows, columns, dtype, block_rows, block_cols, threads, vector, row_threads):
    (matrix_layout, reduced_layout) = _row_layout(block_rows, block_cols, threads, vector, row_threads)
    output_shape = (rows, columns)

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
                    x[i, j] = T.if_then_else(tile * block_cols + j < columns, x[i, j], -T.infinity("float32"))
                T.reduce_max(x, next_maximum, dim=1)
                for i in T.Parallel(block_rows):
                    next_maximum[i] = T.max(maximum[i], next_maximum[i])
                for i, j in T.Parallel(block_rows, block_cols):
                    values[i, j] = T.exp(x[i, j] - next_maximum[i])
                T.reduce_sum(values, partial, dim=1)
                for i in T.Parallel(block_rows):
                    total[i] = total[i] * T.exp(maximum[i] - next_maximum[i]) + partial[i]
                    maximum[i] = next_maximum[i]
            for tile in T.serial(T.ceildiv(columns, block_cols)):
                T.copy(X[bx * block_rows, tile * block_cols], x, coalesced_width=vector)
                for i, j in T.Parallel(block_rows, block_cols):
                    values[i, j] = T.exp(x[i, j] - maximum[i]) / total[i]
                T.copy(values, Y[bx * block_rows, tile * block_cols], coalesced_width=vector)

    return kernel
