import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger


TELEMETRY_KEYS = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "llm_steps",
    "env_steps",
    "tool_calls",
    "token_usage_recorded_steps",
    "token_usage_missing_steps",
]

MEMORY_POOL_KEYS = [
    "memory_count_before",
    "memory_count_after",
    "memory_count_delta",
]

NUMERIC_METRIC_KEYS = TELEMETRY_KEYS + MEMORY_POOL_KEYS

CASE_RESULT_KEYS = [
    "run_id",
    "experiment_name",
    "task_completed",
    "before_score",
    "after_score",
    "uplift_score",
    "task_start_time",
] + NUMERIC_METRIC_KEYS


def _normalize_task_record(data: dict) -> dict:
    record = {
        "run_id": data.get("run_id"),
        "experiment_name": data.get("experiment_name"),
        "task_completed": data.get("task_completed"),
        "before_score": data.get("before_score"),
        "after_score": data.get("after_score"),
        "uplift_score": data.get("uplift_score"),
        "task_start_time": data.get("task_start_time"),
    }
    for key in NUMERIC_METRIC_KEYS:
        record[key] = data.get(key)
    return record


def _iter_result_items(data: dict | list) -> list[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _numeric_values(records: list[dict], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _build_case_rows(file_name: str, task_results: dict[str, list[dict]]) -> list[dict]:
    case_rows: list[dict] = []
    for task_id, records in task_results.items():
        total_attempts = len(records)
        for trial_index, record in enumerate(records, start=1):
            row = {
                "file": file_name,
                "task_id": task_id,
                "trial_index": trial_index,
                "total_attempts": total_attempts,
                "is_final_attempt": trial_index == total_attempts,
            }
            for key in CASE_RESULT_KEYS:
                row[key] = record.get(key)
            case_rows.append(row)
    return case_rows


def _build_task_rows(file_name: str, task_results: dict[str, list[dict]]) -> list[dict]:
    task_rows: list[dict] = []
    for task_id, records in task_results.items():
        if not records:
            continue

        final_record = records[-1]
        final_after_score = final_record.get("after_score")
        final_before_score = final_record.get("before_score")
        final_uplift_score = final_record.get("uplift_score")

        row = {
            "file": file_name,
            "task_id": task_id,
            "num_cases": len(records),
            "final_trial_index": len(records),
            "final_task_completed": final_record.get("task_completed"),
            "final_before_score": float(final_before_score)
            if isinstance(final_before_score, (int, float))
            else None,
            "final_after_score": float(final_after_score)
            if isinstance(final_after_score, (int, float))
            else None,
            "final_uplift_score": float(final_uplift_score)
            if isinstance(final_uplift_score, (int, float))
            else None,
            "pass_final": float(final_after_score >= 1.0)
            if isinstance(final_after_score, (int, float))
            else None,
        }

        final_run_id = final_record.get("run_id")
        if isinstance(final_run_id, (int, float)):
            row["final_run_id"] = int(final_run_id)

        if isinstance(final_record.get("experiment_name"), str):
            row["experiment_name"] = final_record["experiment_name"]

        for metric in NUMERIC_METRIC_KEYS:
            metric_value = final_record.get(metric)
            if isinstance(metric_value, (int, float)):
                row[f"final_{metric}"] = float(metric_value)

        task_rows.append(row)
    return task_rows


def calculate_best_at_k(scores: list, k: int) -> float:
    """
    Calculate best@k
    Divide scores into groups of size k, take the maximum value in each group,
    then average these maximum values

    Args:
        scores: List of after_score values for all runs of a task
        k: Group size

    Returns:
        best@k value
    """
    if len(scores) % k != 0:
        raise ValueError(
            f"Length of scores ({len(scores)}) must be divisible by k ({k})"
        )

    group_maxs = []
    for i in range(0, len(scores), k):
        group = scores[i : i + k]
        group_maxs.append(max(group))

    return sum(group_maxs) / len(group_maxs)


def calculate_pass_at_k(scores: list, k: int) -> float:
    if len(scores) % k != 0:
        raise ValueError(
            f"Length of scores ({len(scores)}) must be divisible by k ({k})"
        )

    group_maxs = []
    for i in range(0, len(scores), k):
        group = scores[i : i + k]
        is_pass = 1.0 if max(group) >= 1.0 else 0.0
        group_maxs.append(is_pass)

    return sum(group_maxs) / len(group_maxs)


def get_possible_k_values(total_runs: int) -> list:
    """
    Get all possible k values (factors of total_runs)

    Args:
        total_runs: Total number of runs

    Returns:
        List of k values in descending order
    """
    k_values = []
    for k in range(1, total_runs + 1):
        if total_runs % k == 0:
            k_values.append(k)
    return sorted(k_values, reverse=True)  # Sort from large to small


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize AppWorld experiment JSONL results"
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model directory name under exp_result (e.g., gpt-oss-120b)",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment file stem without extension (e.g., test_normal_wo_mem)",
    )
    parser.add_argument(
        "--exp-root",
        default="./exp_result",
        help="Root experiment directory containing model subdirectories",
    )
    return parser


def _collect_result_files(
    path: Path, model_name: str | None, experiment_name: str | None
) -> list[Path]:
    if model_name and experiment_name:
        model_file_stem = model_name.replace("/", "__")
        candidates = [
            path / experiment_name / f"{model_file_stem}.jsonl",
            path / model_name / f"{experiment_name}.jsonl",
        ]
        return candidates

    if model_name:
        model_file_stem = model_name.replace("/", "__")
        old_layout = list((path / model_name).glob("*.jsonl"))
        new_layout = list(path.glob(f"*/{model_file_stem}.jsonl"))
        return sorted(set(old_layout + new_layout))

    if experiment_name:
        old_layout = list(path.glob(f"*/{experiment_name}.jsonl"))
        new_layout = list((path / experiment_name).glob("*.jsonl"))
        return sorted(set(old_layout + new_layout))

    # Default: support both old and current layouts
    top_level = list(path.glob("*.jsonl"))
    nested = list(path.glob("*/*.jsonl"))
    return sorted(top_level + nested)


def run_exp_statistic(
    model_name: str | None = None,
    experiment_name: str | None = None,
    exp_root: str = "./exp_result",
):
    path: Path = Path(exp_root)

    # Store results for all experiments
    all_results = {}
    all_case_rows: list[dict] = []
    all_task_rows: list[dict] = []

    target_files = _collect_result_files(
        path=path, model_name=model_name, experiment_name=experiment_name
    )

    if not target_files:
        logger.warning("No matching result files found for given arguments")

    if model_name and not (path / model_name).exists():
        logger.warning(f"Model directory not found: {path / model_name}")

    if (
        model_name
        and experiment_name
        and not any(file.exists() for file in target_files)
    ):
        logger.warning(
            f"Experiment file not found for model={model_name}, experiment={experiment_name}"
        )

    for file in target_files:
        if not file.exists():
            logger.warning(f"File not found: {file}")
            continue

        # Group results by task_id
        task_results = defaultdict(list)

        with open(file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                for item in _iter_result_items(data):
                    if "task_id" not in item:
                        continue
                    task_id = item["task_id"]
                    task_results[task_id].append(_normalize_task_record(item))

        if not task_results:
            logger.warning(f"No valid data found in file {file}")
            continue

        all_case_rows.extend(_build_case_rows(file.name, task_results))
        all_task_rows.extend(_build_task_rows(file.name, task_results))

        attempt_counts = [len(records) for records in task_results.values() if records]
        if not attempt_counts:
            logger.warning(f"No run records found in file {file}")
            continue

        logger.info(
            "File {}: {} tasks, attempt count min={}, max={}",
            file,
            len(task_results),
            min(attempt_counts),
            max(attempt_counts),
        )

        final_records = [records[-1] for records in task_results.values() if records]
        if not final_records:
            logger.warning(f"No final-attempt records found in file {file}")
            continue

        logger.info(
            f"File {file}: final-attempt evaluation on {len(final_records)} tasks"
        )

        # Calculate summary metrics from final attempts only
        file_results: dict[str, str | float] = {"file": file.name}

        for metric in NUMERIC_METRIC_KEYS:
            values = [
                value
                for value in (record.get(metric) for record in final_records)
                if isinstance(value, (int, float))
            ]
            if values:
                file_results[f"avg_{metric}"] = sum(values) / len(values)

        final_scores = [
            score
            for score in (record.get("after_score") for record in final_records)
            if isinstance(score, (int, float))
        ]
        if final_scores:
            avg_best_at_1 = sum(final_scores) / len(final_scores)
            avg_pass_at_1 = sum(
                1.0 if score >= 1.0 else 0.0 for score in final_scores
            ) / len(final_scores)
            file_results["best@1"] = avg_best_at_1
            file_results["pass@1"] = avg_pass_at_1
            logger.info(f"file={file.name} best@1={avg_best_at_1:.4f}")
            logger.info(f"file={file.name} pass@1={avg_pass_at_1:.4f}")

        all_results[file.name] = file_results

    if experiment_name:
        output_dir = path / experiment_name
        summary_output_name = f"experiment_summary_{experiment_name}.csv"
        case_output_name = f"case_results_{experiment_name}.csv"
        task_output_name = f"task_results_{experiment_name}.csv"
    elif model_name:
        output_dir = path / model_name
        summary_output_name = "experiment_summary.csv"
        case_output_name = "case_results.csv"
        task_output_name = "task_results.csv"
    else:
        output_dir = path
        summary_output_name = "experiment_summary.csv"
        case_output_name = "case_results.csv"
        task_output_name = "task_results.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and display summary table
    if all_results:
        df = pd.DataFrame(list(all_results.values()))
        df = df.set_index("file")

        # Sort columns by the number in column name (best@8, best@4, best@2, best@1)
        pass_columns = [col for col in df.columns if col.startswith("pass@")]
        best_columns = [col for col in df.columns if col.startswith("best@")]
        telemetry_columns = [col for col in df.columns if col.startswith("avg_")]
        pass_columns.sort(key=lambda x: x, reverse=False)
        best_columns.sort(key=lambda x: x, reverse=False)
        telemetry_columns.sort()
        ordered_columns = pass_columns + best_columns + telemetry_columns
        if ordered_columns:
            df = df[ordered_columns]

        print("\n" + "=" * 80)
        print("Experiment Results Summary Table")
        print("=" * 80)
        print(df.round(4))
        print("=" * 80)

        summary_output_path = output_dir / summary_output_name
        df.to_csv(summary_output_path)
        logger.info(f"Results table saved to: {summary_output_path}")
    else:
        logger.warning("No valid experiment results found")

    if all_case_rows:
        case_df = pd.DataFrame(all_case_rows)
        case_columns = [
            "file",
            "task_id",
            "trial_index",
            "total_attempts",
            "is_final_attempt",
        ] + CASE_RESULT_KEYS
        ordered_case_columns = [col for col in case_columns if col in case_df.columns]
        if ordered_case_columns:
            case_df = case_df[ordered_case_columns]
        case_output_path = output_dir / case_output_name
        case_df.to_csv(case_output_path, index=False)
        logger.info(f"Case-level results saved to: {case_output_path}")
    else:
        logger.warning("No case-level results found")

    if all_task_rows:
        task_df = pd.DataFrame(all_task_rows)
        base_columns = [
            "file",
            "task_id",
            "experiment_name",
            "num_cases",
            "final_trial_index",
            "final_run_id",
            "final_task_completed",
            "final_before_score",
            "final_after_score",
            "final_uplift_score",
            "pass_final",
        ]
        final_telemetry_columns = sorted(
            [
                col
                for col in task_df.columns
                if col.startswith("final_") and col not in base_columns
            ]
        )
        ordered_task_columns = [
            col for col in base_columns if col in task_df.columns
        ] + final_telemetry_columns
        if ordered_task_columns:
            task_df = task_df[ordered_task_columns]
        task_output_path = output_dir / task_output_name
        task_df.to_csv(task_output_path, index=False)
        logger.info(f"Task-level results saved to: {task_output_path}")
    else:
        logger.warning("No task-level results found")


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_exp_statistic(
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        exp_root=args.exp_root,
    )
