# NSYS_DUMP="tilelang_gemm_advanced_autotune_pipeline_nsys_%n"
NSYS_DUMP="tilelang_gemm_advanced_autotune_multi_gpu_nsys_%n"
# NSYS_DUMP="tilelang_gemm_advanced_autotune_grouped_compile_nsys_%n"


nsys profile \
    --trace=cuda,osrt,python-gil \
    --sample=process-tree \
    --cpuctxsw=process-tree \
    --python-sampling=true \
    --backtrace=fp \
    --force-overwrite=true \
    -o /tmp/${NSYS_DUMP} \
env TILELANG_DISABLE_CACHE=1 TILELANG_AUTO_TUNING_DISABLE_CACHE=1 \
python examples/gemm/example_gemm_advanced_autotune.py --use_autotune --profile_backend event \
    --benchmark_multi_gpu \
    --benchmark_devices 0 1 \
    # --enable_grouped_compile \
