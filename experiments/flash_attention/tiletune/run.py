"""Rank FlashAttention configurations by pipeline_time and print the winner's rank."""

import argparse

from experiments._common import positive_int, prepare_run
from experiments._tiletune import add_arguments, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, "flash_attention")
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--heads", type=positive_int, default=16)
    parser.add_argument("--sequence", type=positive_int, default=4096)
    parser.add_argument("--dim", type=positive_int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--spill-budget-registers-per-thread", type=int, default=32)
    args = parser.parse_args()
    prepare_run(args)

    from examples.flash_attention.example_mha_tiletune import (
        PASS_CONFIGS,
        check_accuracy,
        get_configs,
        make_attention,
        make_inputs,
        reference_attention,
    )

    inputs = make_inputs(args.batch, args.heads, args.sequence, args.dim, args.seed)
    run(
        args,
        kernel=make_attention(args.batch, args.heads, args.sequence, args.dim, args.causal),
        grid=get_configs(),
        inputs=inputs,
        reference=reference_attention(*inputs, args.causal),
        dtype="float16",
        out_idx=[3],
        pass_configs=PASS_CONFIGS,
        check=lambda actuals, refs: check_accuracy(actuals[0], refs[0]),
        options=dict(
            attention_spill_budget_registers_per_thread=args.spill_budget_registers_per_thread,
            max_spill_bytes=None,
            max_local_bytes=None,
        ),
        workload=dict(
            family="flash_attention", batch=args.batch, heads=args.heads, sequence=args.sequence, dim=args.dim, causal=args.causal
        ),
        kernel_source="examples/flash_attention/example_mha_fwd_bshd.py",
    )


if __name__ == "__main__":
    main()
