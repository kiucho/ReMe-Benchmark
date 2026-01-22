#!/bin/bash

SEED=42 # default is 42
NUM_SAMPLES=4 # default is 8

CURR_TIME=$(date +%Y%m%d%H%M%S)
EXPERIMENT_NAME=offline_pool_react_${CURR_TIME}
BENCHMARK_FILE=benchmark_temp.csv
MEMORY_WORKSPACE_ID=nika_temp_jh
AGENT_TYPE=react


python benchmark/run_benchmark.py \
    --mode offline \
    --num-samples $NUM_SAMPLES \
    --agent-type $AGENT_TYPE \
    --memory-workspace-id $MEMORY_WORKSPACE_ID \
    --experiment-name $EXPERIMENT_NAME \
    --seed $SEED \
    --benchmark-file $BENCHMARK_FILE \
    --resume-memory