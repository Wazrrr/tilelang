# CSV_FILE="results/test_auto_grouped_size_1.csv"
# CSV_FILE="results/test_auto_grouped_size_4.csv"

# CSV_FILE="results/test_cython_grouped_size_1.csv"
# CSV_FILE="results/test_cython_grouped_size_4.csv"

# CSV_FILE="results/test_auto_real_grouped_size_4.csv"

# CSV_FILE="results/gemm_autotune_e2e_pipeline.csv"
# CSV_FILE="results/gemm_autotune_e2e_no_pipeline.csv"

# CSV_FILE="results/gemm_autotune_e2e_no_pipeline_all_configs_normal.csv"
CSV_FILE="results/gemm_autotune_e2e_no_pipeline_all_configs_grouped_size_4.csv"

CUDA_VISIBLE_DEVICES=3 python run_gemm_autotune_e2e.py \
  --csv ${CSV_FILE} \
  --cpu-count 16 \
  --execution-backend auto \
  --no-pipeline \
  --grouped-compile \
  --group-compile-size 4