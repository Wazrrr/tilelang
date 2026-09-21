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
    input_values: dict = field(default_factory=dict)

    def check_input_values(self, inputs):
        for index, expected in self.input_values.items():
            actual = inputs[int(index)]
            if actual.ndim != 1 or actual.dtype not in (torch.int32, torch.int64) or actual.tolist() != list(expected):
                raise ValueError(f"input {index} differs from the declared TileTune metadata")

    def check(self, actuals, references):
        """Check both elementwise errors and relative signal error.

        An absolute tolerance alone can accept all-zero softmax/attention output
        when sequence lengths are large. The norm check prevents that failure.
        """
        if len(actuals) != len(references):
            raise AssertionError("wrong number of kernel outputs")
        for actual, reference in zip(actuals, references):
            if actual.dtype != reference.dtype:
                raise AssertionError(f"output dtype {actual.dtype} differs from reference {reference.dtype}")
            # Torch otherwise requests bitwise equality for FP8. Converting the
            # stored values preserves the existing numerical tolerances, shape
            # and device checks, without changing the kernel/output contract.
            if actual.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                torch.testing.assert_close(actual.float(), reference.float(), rtol=self.rtol, atol=self.atol)
            else:
                torch.testing.assert_close(actual, reference, rtol=self.rtol, atol=self.atol)
            error = torch.linalg.vector_norm(actual.float() - reference.float())
            signal = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
            if not torch.isfinite(error) or error / signal > self.rtol:
                raise AssertionError(f"relative output norm error {(error / signal).item()} exceeds {self.rtol}")


def _random(shape, dtype, device, generator):
    # Generate in FP32 so FP8 uses the same seeded input distribution.
    return (torch.rand(shape, device=device, generator=generator) - 0.5).to(getattr(torch, dtype))
