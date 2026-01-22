import csv
import os
from dataclasses import asdict, dataclass

from dotenv import load_dotenv

from nika.utils.session import get_experiment_name

load_dotenv()
RESULTS_DIR = os.getenv("RESULTS_DIR")


@dataclass
class EvalResult:
    agent_type: str = None
    backend_model: str = None
    root_cause_category: str = None
    root_cause_name: str = None
    net_env: str = None
    scenario_topo_size: str = None
    session_id: str = None
    in_tokens: int = None
    out_tokens: int = None
    steps: int = None
    tool_calls: int = None
    tool_errors: int = None
    time_taken: float = None
    llm_judge_relevance_score: int = None
    llm_judge_correctness_score: int = None
    llm_judge_efficiency_score: int = None
    llm_judge_clarity_score: int = None
    llm_judge_final_outcome_score: int = None
    llm_judge_overall_score: int = None
    detection_score: float = None
    localization_accuracy: float = None
    localization_precision: float = None
    localization_recall: float = None
    localization_f1: float = None
    rca_accuracy: float = None
    rca_precision: float = None
    rca_recall: float = None
    rca_f1: float = None
    experience_added_count: int = None
    experience_pool_total_count: int = None


def record_eval_result(eval_result: EvalResult) -> None:
    experiment_name = get_experiment_name()
    if experiment_name:
        log_results_dir = os.path.join(RESULTS_DIR, experiment_name, "0_summary")
    else:
        log_results_dir = os.path.join(RESULTS_DIR, "0_summary")
    os.makedirs(log_results_dir, exist_ok=True)

    log_file_path = os.path.join(log_results_dir, "evaluation_summary.csv")

    data = {
        "exp_name": experiment_name,
        **asdict(eval_result),
    }

    desired_fieldnames = list(data.keys())

    if os.path.exists(log_file_path):
        with open(log_file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

        if existing_fieldnames != desired_fieldnames:
            merged_fieldnames = []
            for key in existing_fieldnames + desired_fieldnames:
                if key not in merged_fieldnames:
                    merged_fieldnames.append(key)

            with open(log_file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow(row)

            desired_fieldnames = merged_fieldnames

    file_exists = os.path.exists(log_file_path)
    with open(log_file_path, "a+", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=desired_fieldnames)
        if not file_exists or os.path.getsize(log_file_path) == 0:
            writer.writeheader()
        writer.writerow(data)
