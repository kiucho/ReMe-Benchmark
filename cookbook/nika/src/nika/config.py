import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_dotenv_path = find_dotenv()
load_dotenv(_dotenv_path or None)


def _resolve_env_path(value: str | None, base_dir: Path) -> str | None:
    if not value:
        return None
    path = Path(os.path.expanduser(value))
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


_env_base_dir = Path(_dotenv_path).resolve().parent if _dotenv_path else Path.cwd()
_default_base_dir = Path(__file__).resolve().parents[2]

BASE_DIR = _resolve_env_path(os.getenv("BASE_DIR"), _env_base_dir) or str(_default_base_dir)

_results_env = os.getenv("RESULTS_DIR")
RESULTS_DIR = _resolve_env_path(_results_env, _env_base_dir) if _results_env else str(Path(BASE_DIR) / "results")

# Optional runtime directory for per-run isolation (e.g., parallel benchmarks).
_runtime_env = os.getenv("RUNTIME_DIR")
RUNTIME_DIR = _resolve_env_path(_runtime_env, _env_base_dir) if _runtime_env else str(Path(BASE_DIR) / "runtime")

# Experiment name for organizing results (can be set via environment variable or programmatically)
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", None)
