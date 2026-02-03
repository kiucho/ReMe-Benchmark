#!/bin/bash

NUM_TRIALS=2 # default is 2
SEED=42 # default is 42

CURR_TIME=$(date +%Y%m%d%H%M%S)
# EXPERIMENT_NAME=w_mem_react_${CURR_TIME}
EXPERIMENT_NAME=w_mem_react_20260124082511
BENCHMARK_FILE=infer_157_v2.csv
MEMORY_WORKSPACE_ID=nika_infer
AGENT_TYPE=react
STARTING_MEMORY_PATH=experience_pool/nika_train_final

python benchmark/run_benchmark.py \
    --mode online \
    --num-trials $NUM_TRIALS \
    --agent-type $AGENT_TYPE \
    --memory-workspace-id $MEMORY_WORKSPACE_ID \
    --experiment-name $EXPERIMENT_NAME \
    --seed $SEED \
    --benchmark-file $BENCHMARK_FILE \
    --resume-memory
    # --starting-memory-path $STARTING_MEMORY_PATH
