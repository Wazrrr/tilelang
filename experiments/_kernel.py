"""Shared numerical checks and seeded inputs for experiment families."""

from dataclasses import dataclass, field
import torch
import tilelang.language as T


@dataclass
class KernelCase:
    build: object
    inputs: object
    reference: object
    out_idx: list[int]
    pass_configs: dict = field(default_factory=dict)
    rtol: float = 0.02
    atol: float = 0.02

    def check(self, actuals, references):
        """Check both elementwise errors and relative signal error.

        An absolute tolerance alone can accept all-zero softmax/attention output
        when sequence lengths are large. The norm check prevents that failure.
        """
        if len(actuals) != len(references):
            raise AssertionError("wrong number of kernel outputs")
        for actual, reference in zip(actuals, references):
            torch.testing.assert_close(actual, reference, rtol=self.rtol, atol=self.atol)
            error = torch.linalg.vector_norm(actual.float() - reference.float())
            signal = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
            if not torch.isfinite(error) or error / signal > self.rtol:
                raise AssertionError(f"relative output norm error {(error / signal).item()} exceeds {self.rtol}")


def _random(shape, dtype, device, generator):
    # Generate in FP32 so FP8 uses the same seeded input distribution.
    return (torch.rand(shape, device=device, generator=generator) - 0.5).to(getattr(torch, dtype))


def positive_integer(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def gemm_policy(name):
    policies = dict(square=T.GemmWarpPolicy.Square, full_row=T.GemmWarpPolicy.FullRow, full_col=T.GemmWarpPolicy.FullCol)
    if name not in policies:
        raise ValueError("warp policy must be square, full_row, or full_col")
    return policies[name]


def _row_layout(block_rows, block_cols, threads, vector, row_threads):
    for name, value in dict(block_rows=block_rows, block_cols=block_cols, threads=threads, vector=vector, row_threads=row_threads).items():
        positive_integer(name, value)
        if value & (value - 1):
            raise ValueError(f"{name} must be a power of two")
    if row_threads > block_rows or threads % row_threads:
        raise ValueError("row thread groups must partition threads and fit block_rows")
    column_threads = threads // row_threads
    if block_cols < column_threads * vector:
        raise ValueError("column tile cannot supply the declared vector/thread layout")
    matrix = T.Fragment(
        (block_rows, block_cols),
        forward_fn=lambda i, j: (
            (i % row_threads) * column_threads + (j // vector) % column_threads,
            (i // row_threads) * (block_cols // column_threads) + (j // (vector * column_threads)) * vector + j % vector,
        ),
    )
    reduced = T.Fragment(
        (block_rows,), forward_fn=lambda i, r: ((i % row_threads) * column_threads + r, i // row_threads), replicate=column_threads
    )
    return matrix, reduced
