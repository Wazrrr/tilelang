from dataclasses import dataclass, field
from typing import Any

import logging

from .base import BaseTemplate
from tvm import te
from ..arch import TileDevice
from ..roller import Hint
from ..config_selector import default_selector_pool_k, select_hints, should_activate_selector, validate_selector_name
from ..utils import get_roller_hints_from_func


logger = logging.getLogger(__name__)


@dataclass
class MatmulTemplate(BaseTemplate):
    """
    A template for matrix multiplication (MatMul).

    This class defines the computation for a matrix-matrix multiplication
    with configurable parameters such as transposition, data types, and bias addition.

    Attributes:
        M (int): Number of rows in matrix A and matrix C.
        N (int): Number of columns in matrix B and matrix C.
        K (int): Number of columns in matrix A and rows in matrix B.
        trans_A (bool): Whether to transpose matrix A before multiplication.
        trans_B (bool): Whether to transpose matrix B before multiplication.
        in_dtype (str): Data type of input matrices.
        out_dtype (str): Data type of output matrix.
        accum_dtype (str): Data type used for accumulation.
        with_bias (bool): Whether to add a bias term.
    """

    # Operation-related configuration parameters
    M: int = None  # Number of rows in matrix A and matrix C
    N: int = None  # Number of columns in matrix B and matrix C
    K: int = None  # Number of columns in matrix A and rows in matrix B
    trans_A: bool = False  # Whether to transpose matrix A
    trans_B: bool = True  # Whether to transpose matrix B
    in_dtype: str = "float16"  # Data type of input matrices
    out_dtype: str = "float16"  # Data type of output matrix
    accum_dtype: str = "float16"  # Data type for accumulation
    with_bias: bool = False  # Whether to add a bias term
    _last_selector_report: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def get_hardware_aware_configs(self, arch: TileDevice = None, topk: int = 10) -> list[Hint]:
        """
        Retrieves optimized hardware-aware configurations.

        Args:
            arch (TileDevice, optional): The target hardware architecture.
            topk (int, optional): Number of top configurations to consider.

        Returns:
            List[Hint]: A list of optimization hints for hardware acceleration.
        """
        roller_hints = get_roller_hints_from_func(self._func, arch=arch, topk=topk, allow_gemv=True)
        return roller_hints

    def recommend_hints(
        self,
        topk: int = 10,
        selector: str = "none",
        selector_pool_k: int | None = None,
        selector_debug: bool = False,
    ) -> list[Hint]:
        """Return hardware-aware hints, optionally reranked by a selector."""
        if topk <= 0:
            self._last_selector_report = {
                "selector_name": "none",
                "selector_input_count": 0,
                "selector_output_count": 0,
                "selector_pool_k": selector_pool_k,
                "selector_time_ms": 0.0,
                "selector_fallback_used": False,
            }
            return []

        validate_selector_name(selector)
        selector_active = should_activate_selector(selector, self._arch, gemm_like=True)
        effective_selector = selector if selector_active else "none"

        pool_k = topk
        if effective_selector != "none":
            pool_k = selector_pool_k if selector_pool_k is not None else default_selector_pool_k(topk)
            if pool_k <= 0:
                raise ValueError("selector_pool_k must be > 0 when provided")

        roller_hints = self.get_hardware_aware_configs(self._arch, topk=pool_k)
        hint_list = list(roller_hints or [])

        if effective_selector == "none":
            self._last_selector_report = {
                "selector_name": "none",
                "selector_input_count": len(hint_list),
                "selector_output_count": len(hint_list),
                "selector_pool_k": selector_pool_k,
                "selector_time_ms": 0.0,
                "selector_fallback_used": False,
            }
            return hint_list

        try:
            selected_hints, telemetry = select_hints(
                hint_list,
                topk=topk,
                selector_name=effective_selector,
                arch=self._arch,
                m=self.M,
                n=self.N,
                k=self.K,
                in_dtype=self.in_dtype,
                accum_dtype=self.accum_dtype,
                selector_pool_k=pool_k,
                selector_debug=selector_debug,
                gemm_like=True,
            )
            self._last_selector_report = telemetry.to_dict()
            return selected_hints
        except Exception as ex:  # pragma: no cover - safety fallback
            logger.warning(
                "Selector '%s' failed (%s: %s). Falling back to Roller top-k order.",
                effective_selector,
                type(ex).__name__,
                ex,
            )
            fallback = hint_list[: min(topk, len(hint_list))]
            self._last_selector_report = {
                "selector_name": effective_selector,
                "selector_input_count": len(hint_list),
                "selector_output_count": len(fallback),
                "selector_pool_k": pool_k,
                "selector_time_ms": 0.0,
                "selector_fallback_used": True,
            }
            return fallback

    def get_last_selector_report(self) -> dict[str, Any]:
        """Return the selector telemetry generated by the previous recommend_hints call."""
        return dict(self._last_selector_report)

    def initialize_function(self) -> None:
        """
        Defines and initializes the matrix multiplication computation.

        This method sets up placeholders for input matrices, computes
        the matrix multiplication using TVM's compute API,
        and optionally applies bias and type casting.

        Raises:
            AssertionError: If M, N, or K are not positive integers.
        """
        M, N, K = self.M, self.N, self.K

        # Ensure M, N, K are valid positive integers
        assert isinstance(M, int) and isinstance(N, int) and isinstance(K, int), "Only Support Integer M, N, K"
        assert M > 0 and N > 0 and K > 0, "M, N, K should be positive"

        # Load configuration parameters
        trans_A, trans_B = self.trans_A, self.trans_B
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype
        with_bias = self.with_bias

        # Define tensor shapes based on transpose flags
        input_shape = (M, K) if not trans_A else (K, M)  # Shape of input matrix A
        weight_shape = (K, N) if not trans_B else (N, K)  # Shape of weight matrix B
        output_shape = (M, N)  # Shape of output matrix C
        Bias_shape = (N,)  # Shape of bias vector

        # Create TVM placeholders for input tensors
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)  # Input matrix A
        B = te.placeholder(weight_shape, name="B", dtype=in_dtype)  # Weight matrix B
        Bias = te.placeholder(Bias_shape, name="Bias", dtype=accum_dtype)  # Bias vector

        # Define a reduction axis for matrix multiplication
        k = te.reduce_axis((0, K), name="k")

        def _compute_matmul(i, j):
            """
            Compute function for matrix multiplication.

            Args:
                i (int): Row index.
                j (int): Column index.

            Returns:
                Computed value for C[i, j] as a sum over the reduction axis.
            """
            A_indices = [i, k] if not trans_A else [k, i]  # Adjust indexing if A is transposed
            B_indices = [k, j] if not trans_B else [j, k]  # Adjust indexing if B is transposed
            return te.sum(A[tuple(A_indices)].astype(accum_dtype) * B[tuple(B_indices)].astype(accum_dtype), axis=k)

        # Compute matrix multiplication result
        C = te.compute(
            output_shape,
            fcompute=_compute_matmul,
            name="C",
        )

        # Optionally apply bias addition
        if with_bias:
            C = te.compute(
                output_shape,
                lambda i, j: C[i, j] + Bias[j],
                name="Bias",
            )

        # Optionally cast the output to a different type
        if out_dtype != accum_dtype:
            C = te.compute(
                output_shape,
                lambda i, j: C[i, j].astype(out_dtype),
                name="D",
            )

        # Set function arguments (including bias if used)
        args = [A, B, Bias, C] if self.with_bias else [A, B, C]
        self.set_function(te.create_prim_func(args))

    def params_as_dict(self):
        """
        Returns the template parameters as a dictionary.

        Returns:
            dict: Dictionary containing template parameter values.
        """
        return {
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "trans_A": self.trans_A,
            "trans_B": self.trans_B,
            "in_dtype": self.in_dtype,
            "out_dtype": self.out_dtype,
            "accum_dtype": self.accum_dtype,
            "with_bias": self.with_bias,
        }

    @property
    def class_attributes(self):
        """
        Returns the class attributes in dictionary form.

        Returns:
            dict: Dictionary of class attributes.
        """
        return self.params_as_dict()

    def __repr__(self) -> str:
        """
        Returns a string representation of the class instance.

        Returns:
            str: A formatted string representation of the class.
        """
        cls_name = self.__class__.__name__
        fields = self.class_attributes
        field_str = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        return f"{cls_name}({field_str})"
