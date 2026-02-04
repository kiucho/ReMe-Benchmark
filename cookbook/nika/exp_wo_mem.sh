#!/bin/bash

CURR_TIME=$(date +%Y%m%d%H%M%S)
RUN_ID=${RUN_ID:-${CURR_TIME}_$$}
NUM_TRIALS=1
EXPERIMENT_NAME=wo_mem_${RUN_ID}
SEED=42
BENCHMARK_FILE=infer_157_p2.csv

python benchmark/run_benchmark.py \
    --mode online_no_memory \
    --num-trials=$NUM_TRIALS \
    --experiment-name=$EXPERIMENT_NAME \
    --seed=$SEED \
    --benchmark-file=$BENCHMARK_FILE
