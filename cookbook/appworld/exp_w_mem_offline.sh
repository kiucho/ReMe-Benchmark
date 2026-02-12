#!/usr/bin/env bash

CURR_TIME=$(date +%Y%m%d%H%M%S)
RUN_ID=${RUN_ID:-${CURR_TIME}_$$}

MODE="w_mem_offline"
BACKEND_MODEL="${BACKEND_MODEL:-${APPWORLD_BACKEND_MODEL:-gpt-oss-120b}}"
DATASET_NAME="${DATASET_NAME:-${APPWORLD_DATASET_NAME:-train}}"
NUM_SAMPLES="${NUM_SAMPLES:-${APPWORLD_NUM_SAMPLES:-4}}"
MAX_WORKERS="${MAX_WORKERS:-${APPWORLD_MAX_WORKERS:-1}}"
BATCH_SIZE="${BATCH_SIZE:-${APPWORLD_BATCH_SIZE:-1}}"

MEMORY_WORKSPACE_ID="${MEMORY_WORKSPACE_ID:-${APPWORLD_MEMORY_WORKSPACE_ID:-appworld_offline}}"
MEMORY_API_URL="${MEMORY_API_URL:-${APPWORLD_MEMORY_API_URL:-http://0.0.0.0:8002/}}"
EXPERIENCE_POOL_DIR="${EXPERIENCE_POOL_DIR:-${APPWORLD_EXPERIENCE_POOL_DIR:-./experience_pool}}"
STARTING_MEMORY_PATH="${STARTING_MEMORY_PATH:-${APPWORLD_STARTING_MEMORY_PATH:-}}"
RESUME_MEMORY="${RESUME_MEMORY:-${APPWORLD_RESUME_MEMORY:-false}}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODE}_${RUN_ID}}"

cmd=(
  python3 run_appworld.py
  --mode "${MODE}"
  --backend-model "${BACKEND_MODEL}"
  --dataset-name "${DATASET_NAME}"
  --experiment-name "${EXPERIMENT_NAME}"
  --num-samples "${NUM_SAMPLES}"
  --max-workers "${MAX_WORKERS}"
  --batch-size "${BATCH_SIZE}"
  --memory-workspace-id "${MEMORY_WORKSPACE_ID}"
  --memory-api-url "${MEMORY_API_URL}"
  --experience-pool-dir "${EXPERIENCE_POOL_DIR}"
)

if [[ -n "${STARTING_MEMORY_PATH}" ]]; then
  cmd+=(--starting-memory-path "${STARTING_MEMORY_PATH}")
fi

if [[ "${RESUME_MEMORY}" == "1" || "${RESUME_MEMORY}" == "true" || "${RESUME_MEMORY}" == "TRUE" ]]; then
  cmd+=(--resume-memory)
fi

"${cmd[@]}"
