#!/bin/bash

NUM_TRIALS=1 # default is 2
SEED=42 # default is 42

CURR_TIME=$(date +%Y%m%d%H%M%S)
EXPERIMENT_NAME=w_mem_react_${CURR_TIME}
BENCHMARK_FILE=benchmark_selected.csv
MEMORY_WORKSPACE_ID=nika_v1
AGENT_TYPE=react
MEMORY_DUMP_PATH=cookbook/nika/experience_pool/nika_test

python benchmark/run_benchmark.py \
    --mode online \
    --num-trials $NUM_TRIALS \
    --agent-type $AGENT_TYPE \
    --memory-workspace-id $MEMORY_WORKSPACE_ID \
    --experiment-name $EXPERIMENT_NAME \
    --seed $SEED \
    --benchmark-file $BENCHMARK_FILE \
    --resume-memory \
    --memory-dump-path $MEMORY_DUMP_PATH
