"""Full-row softmax implementation."""

import tilelang.language as T


def softmax_program(rows, columns, dtype, block_rows, threads):
    width = 1 << (columns - 1).bit_length()
    output_shape = (rows, columns)

    @T.prim_func
    def kernel(X: T.Tensor((rows, columns), dtype), Y: T.Tensor(output_shape, dtype)):
        with T.Kernel(T.ceildiv(rows, block_rows), threads=threads) as bx:
            x = T.alloc_fragment((block_rows, width), "float32")
            values = T.alloc_fragment((block_rows, width), "float32")
            reduced = T.alloc_fragment((block_rows,), "float32")
            T.copy(X[bx * block_rows, 0], x)
            for i, j in T.Parallel(block_rows, width):
                x[i, j] = T.if_then_else(j < columns, x[i, j], -T.infinity("float32"))
            T.reduce_max(x, reduced, dim=1)
            for i, j in T.Parallel(block_rows, width):
                values[i, j] = T.exp(x[i, j] - reduced[i])
            T.reduce_sum(values, reduced, dim=1)
            for i, j in T.Parallel(block_rows, width):
                x[i, j] = values[i, j] / reduced[i]
            T.copy(x, Y[bx * block_rows, 0])

    return kernel
