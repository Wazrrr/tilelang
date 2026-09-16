"""Shared numerical checks and seeded inputs for experiment families."""

from dataclasses import dataclass, field
import torch


@dataclass
class KernelCase:
    build: object
    inputs: object
    reference: object
    out_idx: list[int] | None
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
