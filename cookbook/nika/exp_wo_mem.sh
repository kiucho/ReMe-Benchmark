#!/bin/bash

CURR_TIME=$(date +%Y%m%d%H%M%S)
RUN_ID=${RUN_ID:-${CURR_TIME}_$$}
NUM_TRIALS=1
EXPERIMENT_NAME=wo_mem_${RUN_ID}
SEED=42
BENCHMARK_FILE=test_random_157.csv
BACKEND_MODEL=gpt-oss-120b

python benchmark/run_benchmark.py \
    --mode online_no_memory \
    --backend-model=$BACKEND_MODEL \
    --num-trials=$NUM_TRIALS \
    --experiment-name=$EXPERIMENT_NAME \
    --seed=$SEED \
    --benchmark-file=$BENCHMARK_FILE
