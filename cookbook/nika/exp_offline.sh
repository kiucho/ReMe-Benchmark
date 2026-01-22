#!/bin/bash

SEED=42 # default is 42
NUM_SAMPLES=8 # default is 8

CURR_TIME=$(date +%Y%m%d%H%M%S)
EXPERIMENT_NAME=offline_pool_20260121154816_${CURR_TIME}
BENCHMARK_FILE=benchmark_selected.csv
MEMORY_WORKSPACE_ID=nika_v1


python benchmark/run_benchmark.py \
    --mode offline \
    --memory-workspace-id $MEMORY_WORKSPACE_ID \
    --experiment-name $EXPERIMENT_NAME \
    --seed $SEED \
    --benchmark-file $BENCHMARK_FILE
