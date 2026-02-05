#!/usr/bin/env bash

CURR_TIME=$(date +%Y%m%d%H%M%S)
RUN_ID=${RUN_ID:-${CURR_TIME}_$$}

MODE="wo_mem"
BACKEND_MODEL="${BACKEND_MODEL:-${APPWORLD_BACKEND_MODEL:-gpt-oss-120b}}"
DATASET_NAME="${DATASET_NAME:-${APPWORLD_DATASET_NAME:-test_normal}}"
NUM_TRIALS="${NUM_TRIALS:-${APPWORLD_NUM_TRIALS:-1}}"
MAX_WORKERS="${MAX_WORKERS:-${APPWORLD_MAX_WORKERS:-1}}"
BATCH_SIZE="${BATCH_SIZE:-${APPWORLD_BATCH_SIZE:-1}}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODE}_${RUN_ID}}"

python3 run_appworld.py \
  --mode "${MODE}" \
  --backend-model "${BACKEND_MODEL}" \
  --dataset-name "${DATASET_NAME}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --num-trials "${NUM_TRIALS}" \
  --max-workers "${MAX_WORKERS}" \
  --batch-size "${BATCH_SIZE}"
