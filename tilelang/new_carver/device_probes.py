"""Fixed primitive workloads for reusable CUDA device characterization.

These kernels are independent of candidate kernels/configs. They are compiled
only by the explicit device-profile API, never by PrimFunc analysis.
"""

import tilelang.language as T


CUDA_PRIMITIVES = r"""
template <int Kind, int Iterations>
__device__ __noinline__ float NewCarverPrimitive(int tid) {
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
    name = f"NewCarverPrimitive<{kind}, {iterations}>"

    @T.prim_func
    def main(Out: T.Tensor((blocks, threads), "float32")):
        with T.Kernel(blocks, threads=threads) as bx:
            T.import_source(CUDA_PRIMITIVES)
            tx = T.get_thread_binding()
            Out[bx, tx] = T.call_extern("float32", name, tx)

    return main


def memory_copy(elements):
    @T.prim_func
    def main(A: T.Tensor((elements,), "float32"), Out: T.Tensor((elements,), "float32")):
        with T.Kernel(T.ceildiv(elements, 4096), threads=128) as bx:
            for i in T.Parallel(4096):
                Out[bx * 4096 + i] = A[bx * 4096 + i]

    return main


def tma_roundtrip(iterations, blocks):
    return tile_copy_roundtrip(iterations, blocks, "tma")


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
__device__ __noinline__ float NewCarverReductionPrimitive(int tid) {
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
    name = f"NewCarverReductionPrimitive<{kind}, {iterations}>"

    @T.prim_func
    def main(Out: T.Tensor((blocks, threads), "float32")):
        with T.Kernel(blocks, threads=threads) as bx:
            T.import_source(REDUCTION_PRIMITIVES)
            tx = T.get_thread_binding()
            Out[bx, tx] = T.call_extern("float32", name, tx)

    return main
