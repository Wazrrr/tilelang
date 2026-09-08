"""Rank the FP8 GEMM grid by pipeline_time and print the winner's rank."""

import argparse

from experiments._common import add_gemm_arguments, prepare_run
from experiments._new_carver import add_arguments, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, "gemm_fp8")
    add_gemm_arguments(parser)
    parser.add_argument("--dtype", choices=["float8_e4m3fn", "float8_e5m2"], default="float8_e4m3fn")
    args = parser.parse_args()
    prepare_run(args)

    import torch
    from experiments.gemm_fp8.kernel import get_configs, make_kernel
    from examples.gemm_fp8.example_tilelang_gemm_fp8 import calc_diff

    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    inputs = [
        torch.empty(shape, device="cuda", dtype=torch.float16).uniform_(-0.5, 0.5, generator=generator).to(dtype)
        for shape in ((args.m, args.k), (args.n, args.k))
    ]

    def accuracy(actuals, refs):
        difference = calc_diff(actuals[0], refs[0]).item()
        if not difference < 1e-3:
            raise AssertionError(f"FP8 calc_diff={difference}; expected < 1e-3")

    run(
        args,
        kernel=make_kernel(args.m, args.n, args.k, args.dtype),
        grid=get_configs(),
        inputs=inputs,
        reference=(inputs[0].float() @ inputs[1].float().T).to(dtype),
        dtype=args.dtype,
        out_idx=None,
        check=accuracy,
        workload=dict(family="gemm_fp8", m=args.m, n=args.n, k=args.k, dtype=args.dtype, transpose_b=True),
        kernel_source="examples/gemm_fp8/example_tilelang_gemm_fp8.py",
    )


if __name__ == "__main__":
    main()
