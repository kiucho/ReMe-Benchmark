#!/bin/bash

NUM_TRIALS=2
EXPERIMENT_NAME=w_mem
SEED=42
BENCHMARK_FILE=benchmark_custom.csv

python benchmark/run_benchmark.py \
    --use-memory \
    --num-trials=$NUM_TRIALS \
    --experiment-name=$EXPERIMENT_NAME \
    --seed=$SEED \
    --benchmark-file=$BENCHMARK_FILE
