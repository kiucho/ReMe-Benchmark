#!/usr/bin/env bash

CURR_TIME=$(date +%Y%m%d%H%M%S)
RUN_ID=${RUN_ID:-${CURR_TIME}_$$}

MODE="w_mem_warm"
BACKEND_MODEL="${BACKEND_MODEL:-${APPWORLD_BACKEND_MODEL:-gpt-oss-120b}}"
DATASET_NAME="${DATASET_NAME:-${APPWORLD_DATASET_NAME:-test_normal}}"
NUM_TRIALS="${NUM_TRIALS:-${APPWORLD_NUM_TRIALS:-2}}"
MAX_WORKERS="${MAX_WORKERS:-${APPWORLD_MAX_WORKERS:-1}}"
BATCH_SIZE="${BATCH_SIZE:-${APPWORLD_BATCH_SIZE:-1}}"

MEMORY_WORKSPACE_ID="${MEMORY_WORKSPACE_ID:-${APPWORLD_MEMORY_WORKSPACE_ID:-appworld}}"
MEMORY_API_URL="${MEMORY_API_URL:-${APPWORLD_MEMORY_API_URL:-http://0.0.0.0:8002/}}"
STARTING_MEMORY_PATH="${STARTING_MEMORY_PATH:-${APPWORLD_STARTING_MEMORY_PATH:-}}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODE}_${RUN_ID}}"

if [[ -z "${STARTING_MEMORY_PATH}" ]]; then
  echo "ERROR: warm start requires STARTING_MEMORY_PATH (or APPWORLD_STARTING_MEMORY_PATH)" >&2
  exit 1
fi

python3 run_appworld.py \
  --mode "${MODE}" \
  --backend-model "${BACKEND_MODEL}" \
  --dataset-name "${DATASET_NAME}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --num-trials "${NUM_TRIALS}" \
  --max-workers "${MAX_WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --memory-workspace-id "${MEMORY_WORKSPACE_ID}" \
  --memory-api-url "${MEMORY_API_URL}" \
  --starting-memory-path "${STARTING_MEMORY_PATH}"
