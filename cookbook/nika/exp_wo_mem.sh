#!/bin/bash

NUM_TRIALS=1
EXPERIMENT_NAME=wo_mem
SEED=42
BENCHMARK_FILE=benchmark_custom.csv

python benchmark/run_benchmark.py \
    --num-trials=$NUM_TRIALS \
    --experiment-name=$EXPERIMENT_NAME \
    --seed=$SEED \
    --benchmark-file=$BENCHMARK_FILE
