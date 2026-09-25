"""Fixed primitive workloads for reusable CUDA device characterization.

These kernels are independent of candidate kernels/configs. They are compiled
only by the explicit device-profile API, never by PrimFunc analysis.
"""

import tilelang.language as T


CUDA_PRIMITIVES = r"""
template <int Kind, int Iterations>
__device__ __noinline__ float TileTunePrimitive(int tid) {
  float x[8];
  #pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = 0.01f * (tid + j + 1);
  __shared__ volatile float shared[1024];
  if constexpr (Kind == 1) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) shared[tid + 128 * j] = x[j];
    __syncthreads();
  }
  unsigned long long start = clock64();
  #pragma unroll 1
  for (int i = 0; i < Iterations; ++i) {
    if constexpr (Kind == 4) {
      asm volatile("bar.sync 0;" ::: "memory");
    } else {
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        if constexpr (Kind == 0) {
          asm volatile("fma.rn.f32 %0, %0, %1, %2;"
                       : "+f"(x[j]) : "f"(0.999f), "f"(0.001f));
        } else if constexpr (Kind == 1) {
          x[j] = shared[tid + 128 * j];
          shared[tid + 128 * j] = x[j] + 0.00001f;
        } else if constexpr (Kind == 2) {
          float input = x[j] * 0.01f - 1.0f;
          asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(x[j]) : "f"(input));
        } else if constexpr (Kind == 3) {
          float sum = x[j];
          #pragma unroll
          for (int delta = 16; delta; delta >>= 1) {
            float other;
            asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;"
                         : "=f"(other) : "f"(sum), "r"(delta));
            sum += other;
          }
          x[j] = sum * 0.03125f;
        } else if constexpr (Kind == 5) {
          asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(x[j]));
        } else if constexpr (Kind == 6) {
          float input = x[j] * 0.01f + 2.0f;
          asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(x[j]) : "f"(input));
        }
      }
    }
  }
  if constexpr (Kind == 4) return float(clock64() - start);
  float result = 0;
  #pragma unroll
  for (int j = 0; j < 8; ++j) result += x[j];
  return result;
}
"""


def primitive(kind, iterations, blocks, threads=128):
    if kind in (1, 3, 4) and threads != 128:
        raise ValueError("shared/collective legacy probes require 128 threads")
    name = f"TileTunePrimitive<{kind}, {iterations}>"

    @T.prim_func
    def main(Out: T.Tensor((blocks, threads), "float32")):
        with T.Kernel(blocks, threads=threads) as bx:
            T.import_source(CUDA_PRIMITIVES)
            tx = T.get_thread_binding()
            Out[bx, tx] = T.call_extern("float32", name, tx)

    return main


ASYNC_CLOCKS = r"""
template <int Kind, bool Streaming = false>
__device__ __noinline__ float TileTuneAsyncClocks(const float* input, int tid) {
  __shared__ __align__(16) float storage[4096];
  unsigned long long elapsed = 0;
  #pragma unroll 1
  for (int iteration = 0; iteration < 128; ++iteration) {
    __syncthreads();
    unsigned long long start = clock64();
    #pragma unroll
    for (int j = 0; j < (Kind == 0 ? 8 : 1); ++j) {
      int index = tid * 4 + j * 512;
      unsigned address = unsigned(__cvta_generic_to_shared(storage + index));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                   :: "r"(address), "l"(input + index + (Streaming ? iteration * 512 : 0)) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    unsigned long long issued = clock64();
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    float value = reinterpret_cast<volatile float*>(storage)[tid * 4];
    asm volatile("" :: "f"(value) : "memory");
    unsigned long long ready = clock64();
    elapsed += (Kind == 0 ? issued : ready) - start;
  }
  return float(elapsed) / 128.0f;
}
"""


def async_copy_clocks():
    """Separate copy issue and load-to-use timing without residual subtraction."""

    @T.prim_func
    def main(A: T.Tensor((4096,), "float32"), Out: T.Tensor((128, 2), "float32")):
        with T.Kernel(1, threads=128):
            T.import_source(ASYNC_CLOCKS)
            tx = T.get_thread_binding()
            Out[tx, 0] = T.call_extern("float32", "TileTuneAsyncClocks<0>", T.address_of(A[0]), tx)
            Out[tx, 1] = T.call_extern("float32", "TileTuneAsyncClocks<1>", T.address_of(A[0]), tx)

    return main


def async_copy_streaming_clocks():
    """Read each 2 KB tile once; the caller flushes L2 before each invocation."""

    @T.prim_func
    def main(A: T.Tensor((65536,), "float32"), Out: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            T.import_source(ASYNC_CLOCKS)
            tx = T.get_thread_binding()
            Out[tx] = T.call_extern("float32", "TileTuneAsyncClocks<1, true>", T.address_of(A[0]), tx)

    return main


def memory_copy(elements):
    @T.prim_func
    def main(A: T.Tensor((elements,), "float32"), Out: T.Tensor((elements,), "float32")):
        with T.Kernel(T.ceildiv(elements, 4096), threads=128) as bx:
            for i in T.Parallel(4096):
                Out[bx * 4096 + i] = A[bx * 4096 + i]

    return main


def tile_copy_roundtrip(iterations, blocks, instruction="sync"):
    threads = 128

    @T.prim_func
    def main(A: T.Tensor((blocks, 32, 32), "float32"), Out: T.Tensor((blocks, 32, 32), "float32")):
        with T.Kernel(blocks, threads=threads) as bx:
            shared = T.alloc_shared((32, 32), "float32")
            result = T.alloc_fragment((32, 32), "float32")
            T.clear(result)
            for _ in T.serial(iterations):
                T.copy(A[bx, :, :], shared, prefer_instruction=instruction)
                for i, j in T.Parallel(32, 32):
                    result[i, j] += shared[i, j]
            T.copy(result, Out[bx, :, :])

    return main


def tensor_core(input_dtype, accum_dtype, iterations, blocks, threads):
    @T.prim_func
    def main(A: T.Tensor((64, 128), input_dtype), B: T.Tensor((128, 128), input_dtype), Out: T.Tensor((blocks, 64, 128), accum_dtype)):
        with T.Kernel(blocks, threads=threads) as bx:
            a = T.alloc_shared((64, 128), input_dtype)
            b = T.alloc_shared((128, 128), input_dtype)
            c = T.alloc_fragment((64, 128), accum_dtype)
            T.copy(A, a)
            T.copy(B, b)
            T.clear(c)
            for _ in T.serial(iterations):
                # Keep a true loop-carried dependency with bounded values;
                # repeated positive FP8 accumulation loses small increments.
                T.gemm(a, b, c, transpose_B=True)
                for i, j in T.Parallel(64, 128):
                    c[i, j] = -c[i, j]
            T.copy(c, Out[bx, :, :])

    return main


REDUCTION_PRIMITIVES = r"""
template <int Kind, int Iterations>
__device__ __noinline__ float TileTuneReductionPrimitive(int tid) {
  float x[8];
  #pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = 0.01f * (tid + j + 1);
  #pragma unroll 1
  for (int i = 0; i < Iterations; ++i) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      if constexpr (Kind == 0) {
        asm volatile("add.f32 %0, %0, %1;" : "+f"(x[j]) : "f"(0.0001f));
      } else if constexpr (Kind == 1) {
        asm volatile("max.f32 %0, %0, %1;" : "+f"(x[j]) : "f"(0.5f));
      } else {
        // One physical lane shuffle/combine pair. Sum normalization keeps the
        // dependent sequence finite and is included in its effective rate.
        float value = Kind == 2 ? x[j] * 0.5f : x[j];
        float other;
        asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 31, -1;"
                     : "=f"(other) : "f"(value));
        if constexpr (Kind == 2) {
          asm volatile("add.f32 %0, %1, %2;" : "=f"(x[j]) : "f"(value), "f"(other));
        } else {
          asm volatile("max.f32 %0, %1, %2;" : "=f"(x[j]) : "f"(value), "f"(other));
        }
      }
    }
  }
  float result = 0;
  #pragma unroll
  for (int j = 0; j < 8; ++j) result += x[j];
  return result;
}
"""


def reduction_primitive(kind, iterations, blocks, threads=128):
    """Fixed local-arithmetic or shuffle/combine probe, independent of target tiles."""
    name = f"TileTuneReductionPrimitive<{kind}, {iterations}>"

    @T.prim_func
    def main(Out: T.Tensor((blocks, threads), "float32")):
        with T.Kernel(blocks, threads=threads) as bx:
            T.import_source(REDUCTION_PRIMITIVES)
            tx = T.get_thread_binding()
            Out[bx, tx] = T.call_extern("float32", name, tx)

    return main


HALF_MAX_PRIMITIVES = r"""
#include <cuda_fp16.h>
template <int Kind, int Iterations>
__device__ __noinline__ float TileTuneHalfMaxPrimitive(int tid) {
  unsigned short x[8];
  const unsigned short bound = __half_as_ushort(__float2half(0.5f));
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    x[j] = __half_as_ushort(__float2half(0.01f * (tid + j + 1)));
  }
  #pragma unroll 1
  for (int i = 0; i < Iterations; ++i) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      unsigned short other = bound;
      if constexpr (Kind == 1) {
        unsigned int shuffled;
        asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 31, -1;"
                     : "=r"(shuffled) : "r"(static_cast<unsigned int>(x[j])));
        other = static_cast<unsigned short>(shuffled);
      }
      asm volatile("max.f16 %0, %0, %1;" : "+h"(x[j]) : "h"(other));
    }
  }
  float result = 0;
  #pragma unroll
  for (int j = 0; j < 8; ++j) result += __half2float(__ushort_as_half(x[j]));
  return result;
}
"""


def half_max_primitive(kind, iterations, blocks, threads=128):
    """FP16 local max or shuffle/max pair, matching scalar half reductions."""
    name = f"TileTuneHalfMaxPrimitive<{kind}, {iterations}>"

    @T.prim_func
    def main(Out: T.Tensor((blocks, threads), "float32")):
        with T.Kernel(blocks, threads=threads) as bx:
            T.import_source(HALF_MAX_PRIMITIVES)
            tx = T.get_thread_binding()
            Out[bx, tx] = T.call_extern("float32", name, tx)

    return main


FP8_CONVERSION = r"""
template <int Kind, int Iterations>
__device__ __noinline__ float TileTuneFp8Conversion() {
  unsigned short packed[8];
  const float value = 0.125f;
  #pragma unroll 1
  for (int iteration = 0; iteration < Iterations; ++iteration) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      if constexpr (Kind == 0) {
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                     : "=h"(packed[j]) : "f"(value), "f"(value));
      } else {
        asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
                     : "=h"(packed[j]) : "f"(value), "f"(value));
      }
    }
  }
  unsigned int result = 0;
  #pragma unroll
  for (int j = 0; j < 8; ++j) result += packed[j];
  return float(result);
}
"""


def fp8_conversion(dtype, iterations, blocks, threads=128):
    if str(dtype) not in ("float8_e4m3fn", "float8_e5m2"):
        raise ValueError("FP8 conversion probe requires E4M3FN or E5M2")
    kind = 0 if str(dtype) == "float8_e4m3fn" else 1
    name = f"TileTuneFp8Conversion<{kind}, {iterations}>"

    @T.prim_func
    def main(Out: T.Tensor((blocks, threads), "float32")):
        with T.Kernel(blocks, threads=threads) as bx:
            T.import_source(FP8_CONVERSION)
            tx = T.get_thread_binding()
            Out[bx, tx] = T.call_extern("float32", name)

    return main
