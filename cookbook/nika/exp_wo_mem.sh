#!/bin/bash

CURR_TIME=$(date +%Y%m%d%H%M%S)
RUN_ID=${RUN_ID:-${CURR_TIME}_$$}
NUM_TRIALS=1
EXPERIMENT_NAME=wo_mem_${RUN_ID}
SEED=42
BENCHMARK_FILE=infer_157_p2.csv
RUNTIME_DIR=${RUNTIME_DIR:-$(pwd)/runtime_${RUN_ID}}
LAB_NAME_SUFFIX=${LAB_NAME_SUFFIX:-${RUN_ID}}

RUNTIME_DIR=$RUNTIME_DIR python benchmark/run_benchmark.py \
    --mode online_no_memory \
    --num-trials=$NUM_TRIALS \
    --experiment-name=$EXPERIMENT_NAME \
    --lab-name-suffix=$LAB_NAME_SUFFIX \
    --no-wipe-kathara \
    --seed=$SEED \
    --benchmark-file=$BENCHMARK_FILE
