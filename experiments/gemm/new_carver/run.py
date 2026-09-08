"""Rank the advanced-autotune GEMM grid by pipeline_time and print the winner's rank."""

import argparse

from experiments._common import add_gemm_arguments, prepare_run
from experiments._new_carver import add_arguments, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, "gemm")
    add_gemm_arguments(parser)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    args = parser.parse_args()
    prepare_run(args)

    from experiments.gemm.kernel import get_configs, make_inputs, make_kernel, reference

    inputs = make_inputs(args.m, args.n, args.k, args.dtype, args.seed)
    run(
        args,
        kernel=make_kernel(args.m, args.n, args.k, args.dtype),
        grid=get_configs(),
        inputs=inputs,
        reference=reference(*inputs),
        dtype=args.dtype,
        out_idx=[2],
        workload=dict(family="gemm", m=args.m, n=args.n, k=args.k, dtype=args.dtype, transpose_b=True),
        kernel_source="experiments/gemm/kernel.py",
    )


if __name__ == "__main__":
    main()
