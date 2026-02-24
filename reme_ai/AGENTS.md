# REME_AI KNOWLEDGE BASE

## OVERVIEW

`reme_ai/` is the installable core memory package. Keep this tree reusable and benchmark-agnostic.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CLI/app bootstrap | `reme_ai/main.py`, `reme_ai/core/main.py` | service startup path |
| Core context and flow | `reme_ai/core/context/`, `reme_ai/core/flow/` | runtime context wiring |
| LLM + embedding adapters | `reme_ai/core/llm/`, `reme_ai/core/embedding/` | model abstraction layer |
| Memory schemas | `reme_ai/schema/`, `reme_ai/core/schema/` | domain object definitions |
| Retrieval pipeline | `reme_ai/retrieve/` | task/personal/tool/working retrieval ops |
| Summary pipeline | `reme_ai/summary/` | task/personal/tool/working summary ops |
| Vector store backends | `reme_ai/vector_store/`, `reme_ai/core/vector_store/` | storage/update/search |
| Agent integrations | `reme_ai/agent/`, `reme_ai/mem_agent/`, `reme_ai/mem_tool/` | tool + react style integration |

## LOCAL CONVENTIONS

- Preserve package boundaries: reusable logic belongs here, not under `cookbook/`.
- Prefer `loguru` logging style already used in this tree.
- Keep async boundaries explicit; avoid burying event-loop control inside shared ops.
- Follow existing op naming (`*_op.py`) and memory-type partitioning (task/personal/tool/working).

## ANTI-PATTERNS (LOCAL)

- Do not import benchmark-specific code from `cookbook/*` into `reme_ai/*`.
- Do not hardcode experiment paths or result folders in core package modules.
- Do not duplicate vector-store logic across `core/vector_store` and `vector_store`; extend existing abstractions.
- Do not add ad hoc script behavior to import-time package code.

## COMMANDS

```bash
# from repo root
source .venv-reme/bin/activate
python -m build
pytest -q
python tests/test_reme.py
python tests/test_tool.py
python tests/test_vector_store.py
```
