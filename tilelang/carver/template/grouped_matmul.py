"""Grouped matrix multiplication as independent, possibly ragged GEMMs.

The mathematical graph preserves each group's M and B layout. The caller must
state how independent graph costs are combined for its packed kernel launch.
"""

from dataclasses import dataclass, field
from .base import BaseTemplate
from .matmul import MatmulTemplate


@dataclass
class ExplicitArchMatmul(MatmulTemplate):
    _arch: object = field(default=None, repr=False)


@dataclass
class GroupedMatmulTemplate(BaseTemplate):
    _arch: object = field(default=None, repr=False)
    batch_sizes: tuple = ()
    N: int = 1
    K: int = 1
    trans_B: bool = False
    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float32"

    def initialize_function(self):
        if not self.batch_sizes or any(type(m) is not int or m <= 0 for m in self.batch_sizes):
            raise ValueError("grouped matmul requires positive group sizes")
        self.groups = [
            ExplicitArchMatmul(
                M=m,
                N=self.N,
                K=self.K,
                trans_B=self.trans_B,
                in_dtype=self.in_dtype,
                out_dtype=self.out_dtype,
                accum_dtype=self.accum_dtype,
                _arch=self.arch,
            )
            for m in self.batch_sizes
        ]
        self.set_function([group.equivalent_function() for group in self.groups])

    def get_hardware_aware_configs(self, arch=None, topk=10):
        return [group.get_hardware_aware_configs(arch or self.arch, topk) for group in self.groups]
