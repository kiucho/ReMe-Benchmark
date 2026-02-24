#!/usr/bin/env bash

BACKEND_MODEL="gpt-oss-120b"
DATASET_NAME="train"
NUM_SAMPLES="8"
MAX_WORKERS="1"
BATCH_SIZE="1"
MEMORY_API_URL="http://0.0.0.0:8002/"
EXPERIENCE_POOL_DIR="./experience_pool"

CURR_TIME=$(date +%Y%m%d%H%M%S)
RUN_ID=${RUN_ID:-${CURR_TIME}_$$}
MODE="w_mem_offline"

MEMORY_WORKSPACE_ID="${BACKEND_MODEL}_${DATASET_NAME}_n${NUM_SAMPLES}"
EXPERIMENT_NAME="${MODE}_${RUN_ID}"

python3 run_appworld.py \
  --mode "${MODE}" \
  --backend-model "${BACKEND_MODEL}" \
  --dataset-name "${DATASET_NAME}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --num-samples "${NUM_SAMPLES}" \
  --max-workers "${MAX_WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --memory-workspace-id "${MEMORY_WORKSPACE_ID}" \
  --memory-api-url "${MEMORY_API_URL}" \
  --experience-pool-dir "${EXPERIENCE_POOL_DIR}"