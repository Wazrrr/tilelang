"""Token-parallel KDA intra-chunk coefficient construction.

Each CTA owns one token and a tile of heads. It constructs the causal
query/key coefficients for the enclosing chunk and the beta-weighted key/key
coefficients for the enclosing sub-chunk. This is the KDA-specific intra stage;
unlike ``chunk_o.py`` it does not mix a prepared state with values.
"""

import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune


def prepare_input(B, S, H, DK, chunk_size, input_dtype, output_dtype, accum_dtype, gate_dtype):
    """Create the inputs used by the standalone correctness/performance demo."""
    del output_dtype, accum_dtype
    import torch
    import torch.nn.functional as F

    q = torch.randn(B, S, H, DK, dtype=input_dtype, device="cuda")
    k = torch.randn(B, S, H, DK, dtype=input_dtype, device="cuda")
    beta = torch.randn(B, S, H, dtype=input_dtype, device="cuda").sigmoid()
    gates = F.logsigmoid(torch.randn(B, S, H, DK, dtype=gate_dtype, device="cuda"))
    gk = gates.reshape(B, S // chunk_size, chunk_size, H, DK).cumsum(2).reshape(B, S, H, DK)
    return q, k, gk, beta


def prepare_output(B, S, H, chunk_size, sub_chunk_size, output_dtype):
    """Allocate zeroed coefficient tensors for the sparse intra blocks."""
    import torch

    aqk = torch.zeros(B, S, H, chunk_size, dtype=output_dtype, device="cuda")
    akk = torch.zeros(B, S, H, sub_chunk_size, dtype=output_dtype, device="cuda")
    return aqk, akk


def get_configs():
    """Return the example's native autotuning grid."""
    import itertools

    return [
        {"block_H": block_h, "threads": threads, "num_stages": stages}
        for block_h, threads, stages in itertools.product([1, 2, 4, 8], [128, 256], [0, 1, 2, 3])
    ]


@autotune(configs=get_configs(), warmup=3, rep=5)
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_chunk_kda_fwd_intra_token_parallel(
    B,
    S,
    H,
    DK,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    chunk_size,
    sub_chunk_size,
    scale,
    block_H=1,
    threads=128,
    num_stages=1,
):
    """Build the fixed-length BSHD token-parallel KDA intra kernel."""
    assert S % chunk_size == 0, "sequence length must contain complete chunks"
    assert chunk_size % sub_chunk_size == 0, "chunk size must contain complete sub-chunks"
    cs, scs = chunk_size, sub_chunk_size
    q_shape = (B, S, H, DK)
    k_shape = (B, S, H, DK)
    gk_shape = (B, S, H, DK)
    beta_shape = (B, S, H)
    aqk_shape = (B, S, H, cs)
    akk_shape = (B, S, H, scs)

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype=input_dtype),
        K: T.Tensor(k_shape, dtype=input_dtype),
        GK: T.Tensor(gk_shape, dtype=gate_dtype),
        Beta: T.Tensor(beta_shape, dtype=input_dtype),
        Aqk: T.Tensor(aqk_shape, dtype=output_dtype),
        Akk: T.Tensor(akk_shape, dtype=output_dtype),
    ):
        with T.Kernel(B * S, T.ceildiv(H, block_H), threads=threads) as (bbs, bh):
            bb, bs = bbs // S, bbs % S
            chunk = bs // cs
            sub_chunk = (bs % cs) // scs
            chunk_start = chunk * cs
            sub_chunk_start = chunk_start + sub_chunk * scs
            iterations = bs + 1 - sub_chunk_start

            q_i_shared = T.alloc_shared((block_H, DK), dtype=input_dtype)
            k_i_shared = T.alloc_shared((block_H, DK), dtype=input_dtype)
            g_i_shared = T.alloc_shared((block_H, DK), dtype=gate_dtype)
            beta_shared = T.alloc_shared((block_H,), dtype=input_dtype)
            k_j_shared = T.alloc_shared((block_H, DK), dtype=input_dtype)
            g_j_shared = T.alloc_shared((block_H, DK), dtype=gate_dtype)
            aqk_product_shared = T.alloc_shared((block_H, DK), dtype=accum_dtype)
            akk_product_shared = T.alloc_shared((block_H, DK), dtype=accum_dtype)
            aqk_shared = T.alloc_shared((block_H, cs), dtype=output_dtype)
            akk_shared = T.alloc_shared((block_H, scs), dtype=output_dtype)

            q_i_fragment = T.alloc_fragment((block_H, DK), dtype=accum_dtype)
            k_i_fragment = T.alloc_fragment((block_H, DK), dtype=accum_dtype)
            k_j_fragment = T.alloc_fragment((block_H, DK), dtype=accum_dtype)
            aqk_sum = T.alloc_fragment((block_H,), dtype=accum_dtype)
            akk_sum = T.alloc_fragment((block_H,), dtype=accum_dtype)

            T.copy(Q[bb, bs, bh * block_H : (bh + 1) * block_H, :], q_i_shared)
            T.copy(K[bb, bs, bh * block_H : (bh + 1) * block_H, :], k_i_shared)
            T.copy(GK[bb, bs, bh * block_H : (bh + 1) * block_H, :], g_i_shared)
            for i_h in T.Parallel(block_H):
                global_h = bh * block_H + i_h
                beta_shared[i_h] = T.if_then_else(global_h < H, Beta[bb, bs, global_h], 0)

            for i_h, i_k in T.Parallel(block_H, DK):
                q_i_fragment[i_h, i_k] = q_i_shared[i_h, i_k] * scale
                k_i_fragment[i_h, i_k] = k_i_shared[i_h, i_k] * beta_shared[i_h]

            T.clear(aqk_shared)
            T.clear(akk_shared)
            for offset in T.Pipelined(iterations, num_stages=num_stages):
                j = offset + sub_chunk_start
                T.copy(K[bb, j, bh * block_H : (bh + 1) * block_H, :], k_j_shared)
                T.copy(GK[bb, j, bh * block_H : (bh + 1) * block_H, :], g_j_shared)
                for i_h, i_k in T.Parallel(block_H, DK):
                    k_j_fragment[i_h, i_k] = k_j_shared[i_h, i_k] * T.exp2(
                        g_i_shared[i_h, i_k] - g_j_shared[i_h, i_k]
                    )
                    aqk_product_shared[i_h, i_k] = q_i_fragment[i_h, i_k] * k_j_fragment[i_h, i_k]
                    akk_product_shared[i_h, i_k] = k_i_fragment[i_h, i_k] * k_j_fragment[i_h, i_k]

                T.reduce_sum(aqk_product_shared, aqk_sum, dim=-1, clear=True)
                T.reduce_sum(akk_product_shared, akk_sum, dim=-1, clear=True)
                T.copy(aqk_sum, aqk_shared[:, j % cs])
                for i_h in T.Parallel(block_H):
                    akk_shared[i_h, offset] = T.if_then_else(j < bs, akk_sum[i_h], 0)

            T.copy(aqk_shared, Aqk[bb, bs, bh * block_H : (bh + 1) * block_H, :])
            T.copy(akk_shared, Akk[bb, bs, bh * block_H : (bh + 1) * block_H, :])

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    scale,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    chunk_size,
    sub_chunk_size,
):
    """Compare the example against the optional FLA Triton implementation."""
    import torch

    if __package__:
        from .FLA_KDA.fla_chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
        from .test_utils_kda import do_bench
    else:
        from FLA_KDA.fla_chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
        from test_utils_kda import do_bench

    q, k, gk, beta = prepare_input(
        B,
        S,
        H,
        DK,
        chunk_size,
        getattr(torch, input_dtype),
        getattr(torch, output_dtype),
        getattr(torch, accum_dtype),
        getattr(torch, gate_dtype),
    )
    aqk_ref, akk_ref = prepare_output(B, S, H, chunk_size, sub_chunk_size, getattr(torch, output_dtype))
    aqk_ref, akk_ref = chunk_kda_fwd_intra_token_parallel(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        Aqk=aqk_ref,
        Akk=akk_ref,
        scale=scale,
        chunk_size=chunk_size,
        sub_chunk_size=sub_chunk_size,
    )
    kernel = tilelang_chunk_kda_fwd_intra_token_parallel(
        B,
        S,
        H,
        DK,
        input_dtype,
        output_dtype,
        accum_dtype,
        gate_dtype,
        chunk_size,
        sub_chunk_size,
        scale,
    )
    aqk_tilelang, akk_tilelang = kernel(q, k, gk, beta)
    torch.testing.assert_close(aqk_tilelang, aqk_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(akk_tilelang, akk_ref, rtol=2e-2, atol=2e-2)
    print(
        "fla time:",
        do_bench(
            chunk_kda_fwd_intra_token_parallel,
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=aqk_ref,
            Akk=akk_ref,
            scale=scale,
            chunk_size=chunk_size,
            sub_chunk_size=sub_chunk_size,
        ),
        "ms",
    )
    print("tilelang time:", do_bench(kernel, q, k, gk, beta), "ms")


def main():
    import torch

    torch.random.manual_seed(42)
    run_test(
        B=1,
        S=8192,
        H=64,
        DK=128,
        scale=128**-0.5,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        chunk_size=64,
        sub_chunk_size=16,
    )


if __name__ == "__main__":
    main()
