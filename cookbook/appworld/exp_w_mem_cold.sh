#!/usr/bin/env bash

BACKEND_MODEL="gpt-oss-120b"
DATASET_NAME="test_normal"
NUM_TRIALS="2"
MAX_WORKERS="1"
BATCH_SIZE="1"
MEMORY_WORKSPACE_ID="appworld"
MEMORY_API_URL="http://0.0.0.0:8002/"

CURR_TIME=$(date +%Y%m%d%H%M%S)
RUN_ID=${RUN_ID:-${CURR_TIME}_$$}
MODE="w_mem_cold"

EXPERIMENT_NAME="${MODE}_${RUN_ID}"

python3 run_appworld.py \
  --mode "${MODE}" \
  --backend-model "${BACKEND_MODEL}" \
  --dataset-name "${DATASET_NAME}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --num-trials "${NUM_TRIALS}" \
  --max-workers "${MAX_WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --memory-workspace-id "${MEMORY_WORKSPACE_ID}" \
  --memory-api-url "${MEMORY_API_URL}"
