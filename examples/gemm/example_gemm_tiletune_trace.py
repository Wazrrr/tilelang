"""Write a readable GEMM analysis trace without compiling or running a kernel.

Run from the repository root:
    python -m examples.gemm.example_gemm_tiletune_trace --output /tmp/gemm_trace.log

The PrimFunc is real. The device limits and rates below are illustrative inputs,
not measured hardware costs. They make each timing calculation easy to follow.
"""

import argparse

import tilelang.language as T
from tilelang.tiletune import analyze_prim_func


ILLUSTRATIVE_LIMITS = dict(
    sm_count=4,
    shared_memory_per_sm=233472,
    shared_memory_per_block=232448,
    registers_per_sm=65536,
    max_threads_per_sm=2048,
    max_threads_per_block=1024,
    max_blocks_per_sm=32,
    warp_size=32,
)
ILLUSTRATIVE_PROFILE = dict(
    global_bytes_per_cycle=64,
    shared_bytes_per_cycle=128,
    gemm_flops_per_cycle=2048,
    wgmma_flops_per_cycle_per_warpgroup=1024,
    elementwise_ops_per_cycle=128,
    copy_latency_cycles=400,
    barrier_cycles=16,
)


def make_gemm(block_m=64, block_n=64, block_k=32, stages=2, threads=128):
    @T.prim_func
    def main(A: T.Tensor((256, 256), "float16"), B: T.Tensor((256, 256), "float16"), C: T.Tensor((256, 256), "float32")):
        with T.Kernel(T.ceildiv(256, block_m), T.ceildiv(256, block_n), threads=threads) as (bx, by):
            a_shared = T.alloc_shared((block_m, block_k), "float16")
            b_shared = T.alloc_shared((block_k, block_n), "float16")
            c_fragment = T.alloc_fragment((block_m, block_n), "float32")
            T.clear(c_fragment)
            for k in T.Pipelined(T.ceildiv(256, block_k), num_stages=stages):
                T.copy(A[bx * block_m, k * block_k], a_shared)
                T.copy(B[k * block_k, by * block_n], b_shared)
                T.gemm(a_shared, b_shared, c_fragment)
            T.copy(c_fragment, C[bx * block_m, by * block_n])

    return main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/gemm_trace.log", help="Append one analysis block to this file")
    args = parser.parse_args()
    config = dict(block_m=64, block_n=64, block_k=32, stages=2, threads=128)
    result = analyze_prim_func(
        make_gemm(**config),
        dict(trace_path=args.output, ranking_metric="pipeline_time", performance_model=ILLUSTRATIVE_PROFILE),
        target={"kind": "cuda", "arch": "sm_90a"},
        device_limits=ILLUSTRATIVE_LIMITS,
        trace_context={"config": config, "hardware_inputs": "illustrative constants, not profiled measurements"},
    )
    print(f"Trace appended to {args.output}")
    print(f"Accumulator lower bound: {result['pressure']['modeled_lower_bound']} registers/thread")
    print(f"Illustrative pipeline score: {result['tile_cost']['score']} cycles")


if __name__ == "__main__":
    main()
