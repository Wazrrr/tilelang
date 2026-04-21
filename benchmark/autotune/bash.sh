# CSV_FILE="results/gemm_autotune_e2e_pipeline.csv"
CSV_FILE="results/gemm_autotune_e2e_no_pipeline.csv"

CUDA_VISIBLE_DEVICES=4 python run_gemm_autotune_e2e.py \
  --cpu-count 16 \
  --with-roller \
  --csv ${CSV_FILE}