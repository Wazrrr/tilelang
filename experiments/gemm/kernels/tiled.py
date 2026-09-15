"""Family-owned portable implementations and independent references."""

import tilelang.language as T
from experiments._kernel import KernelCase, _random
from experiments.gemm.reference import suite_reference


def gemm_case(w):
    p, dtype = w.parameters, w.dtype
    batch, m, n, k = p.get("batch", 1), p["m"], p["n"], p["k"]
    ta, tb, epilogue = p.get("transpose_a", False), p.get("transpose_b", False), p.get("epilogue", "none")
    ashape, bshape = (batch, k, m) if ta else (batch, m, k), (batch, n, k) if tb else (batch, k, n)
    output_dtype = "float16" if dtype.startswith("float8") else dtype

    def build(block_m, block_n, block_k, stages, threads, warp_policy="square", swizzle_panel=0):
        from experiments._kernel import gemm_policy, positive_integer

        policy = gemm_policy(warp_policy)
        for key, value in dict(block_m=block_m, block_n=block_n, block_k=block_k, threads=threads).items():
            positive_integer(key, value)
        if type(stages) is not int or stages < 0 or type(swizzle_panel) is not int or swizzle_panel < 0:
            raise ValueError("stages and swizzle_panel must be nonnegative integers")
        ashape = (batch, k, m) if ta else (batch, m, k)
        bshape = (batch, n, k) if tb else (batch, k, n)

        @T.prim_func
        def kernel(
            A: T.Tensor(ashape, dtype),
            B: T.Tensor(bshape, dtype),
            Bias: T.Tensor((n,), output_dtype),
            C: T.Tensor((batch, m, n), output_dtype),
        ):
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), batch, threads=threads) as (bx, by, bz):
                T.use_swizzle(panel_size=swizzle_panel, enable=swizzle_panel > 0)
                a = T.alloc_shared((block_k, block_m) if ta else (block_m, block_k), dtype)
                b = T.alloc_shared((block_n, block_k) if tb else (block_k, block_n), dtype)
                c = T.alloc_fragment((block_m, block_n), "float32")
                T.clear(c)
                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=stages):
                    if ta:
                        T.copy(A[bz, kk * block_k, by * block_m], a)
                    else:
                        T.copy(A[bz, by * block_m, kk * block_k], a)
                    if tb:
                        T.copy(B[bz, bx * block_n, kk * block_k], b)
                    else:
                        T.copy(B[bz, kk * block_k, bx * block_n], b)
                    T.gemm(a, b, c, transpose_A=ta, transpose_B=tb, policy=policy)
                if epilogue != "none":
                    for i, j in T.Parallel(block_m, block_n):
                        if bx * block_n + j < n:
                            c[i, j] += Bias[bx * block_n + j]
                    if epilogue == "bias_relu":
                        for i, j in T.Parallel(block_m, block_n):
                            c[i, j] = T.max(c[i, j], 0)
                T.copy(c, C[bz, by * block_m, bx * block_n])

        return kernel

    def inputs(device, generator):
        return [
            _random(ashape, dtype, device, generator),
            _random(bshape, dtype, device, generator),
            _random((n,), output_dtype, device, generator),
        ]

    return KernelCase(build, inputs, suite_reference(w), [3])
