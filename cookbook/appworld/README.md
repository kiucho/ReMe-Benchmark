# AppWorld Benchmark (ReAct Agent)

This folder contains a minimal ReAct-style agent runner for AppWorld tasks.

## Setup

Use `uv` to create a virtualenv and install dependencies:

```bash
cd cookbook/appworld

# NOTE: `appworld` currently depends on `uvloop`, which may not provide wheels for Python 3.13.
# Use Python 3.12 for this benchmark environment.
uv venv --python 3.12 .venv-appworld
source .venv-appworld/bin/activate

uv pip install -r requirements-appworld.txt
```

## Model configuration (GPT-OSS 120B)

The agent supports routing OpenAI-compatible requests to the same GPT-OSS 120B backend configuration used in `cookbook/nika/`.

Set these environment variables (recommended: put them in the repo-root `.env`, which this runner loads):

```bash
# required for GPT-OSS backend
export GPT_OSS_API_URL="https://<your-gpt-oss-endpoint>"
export GPT_OSS_API_KEY="<your-gpt-oss-api-key>"

# optional
export GPT_OSS_MODEL_ID="kt-gpt-oss-rh014"   # default if unset
export GPT_OSS_VERIFY_SSL="false"           # default is false (matches NIKA)

# pick the backend model for AppWorld
export APPWORLD_BACKEND_MODEL="gpt-oss-120b"
```

Notes:
- When `APPWORLD_BACKEND_MODEL` starts with `gpt-oss-120b`, the agent translates it to `GPT_OSS_MODEL_ID` (default: `kt-gpt-oss-rh014`) and sends requests to `GPT_OSS_API_URL` using `GPT_OSS_API_KEY`.

## Memory server (optional)

If you run with `use_memory=True`, you must have the ReMe memory server running (default URL in code: `http://0.0.0.0:8002/`).

## Run

Run from this directory so the relative `.env` path resolves correctly:

```bash
cd cookbook/appworld

# warm start with an existing starting pool
python3 run_appworld.py --mode w_mem_warm --starting-memory-path /abs/path/to/memory_dump

# cold start with memory (empty workspace; accumulate online)
python3 run_appworld.py --mode w_mem_cold

# no memory
python3 run_appworld.py --mode wo_mem
```

## Preset scripts

For convenience, you can run the three common modes via shell scripts (these will create/use `.venv-appworld` via `uv`):

```bash
cd cookbook/appworld

./exp_wo_mem.sh
./exp_w_mem_cold.sh
./exp_w_mem_warm.sh
```

Mode definitions:
- `wo_mem`: no memory calls
- `w_mem_cold`: start from an empty workspace and accumulate memories sequentially
- `w_mem_warm`: load a provided starting pool, then keep accumulating

## Common arguments

Most knobs are exposed as CLI arguments in `cookbook/appworld/run_appworld.py`:

Note: for memory benchmarks, run sequentially with `--max-workers 1 --batch-size 1` (these are the defaults).

```bash
python3 run_appworld.py \
  --mode w_mem_warm \
  --backend-model gpt-oss-120b \
  --dataset-name test_normal \
  --experiment-name w_mem_warm_20260205123456 \
  --num-trials 2 \
  --max-workers 1 \
  --batch-size 1
```
