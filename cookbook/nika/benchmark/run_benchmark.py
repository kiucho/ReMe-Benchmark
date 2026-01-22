import argparse
import csv
import json
import os
import time

import polars as pl
import requests
from tqdm import tqdm

from nika.config import RESULTS_DIR
from nika.evaluator.result_log import record_eval_result
from nika.net_env.net_env_pool import get_net_env_instance
from nika.orchestrator.tasks.detection import DetectionSubmission
from nika.orchestrator.tasks.localization import LocalizationTask
from nika.orchestrator.tasks.rca import RCATask
from nika.utils.session import set_experiment_name
from scripts.step1_net_env_start import start_net_env
from scripts.step2_failure_inject import inject_failure
from scripts.step3_agent_run import start_agent
from scripts.step4_result_eval import eval_results

cur_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXPERIENCE_POOL_DIR = os.path.join(os.path.dirname(cur_dir), "experience_pool")
CASE_LOG_FIELDS = [
    "timestamp",
    "problem",
    "scenario",
    "topo_size",
    "status",
    "completed_trials",
    "num_trials",
    "error_stage",
    "error_type",
    "error_message",
    "detection_score",
    "loc_acc",
    "rca_acc",
    "llm_judge_score",
]


def init_benchmark_case_log(experiment_name: str | None) -> str:
    base_dir = os.path.join(RESULTS_DIR, experiment_name) if experiment_name else RESULTS_DIR
    os.makedirs(base_dir, exist_ok=True)
    log_path = os.path.join(base_dir, "benchmark_case_status.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CASE_LOG_FIELDS)
            writer.writeheader()
    return log_path


def append_benchmark_case_log(log_path: str, row: dict) -> None:
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_LOG_FIELDS)
        writer.writerow(row)


def wipe_kathara():
    """Wipe all Kathara resources to ensure clean state."""
    try:
        from Kathara.manager.Kathara import Kathara
        Kathara.get_instance().wipe()
        print("Kathara wiped successfully")
    except Exception as e:
        print(f"Warning: Kathara wipe failed: {e}")


# ================== ReMe Memory API Helper Functions ==================


def handle_api_response(response: requests.Response):
    """Handle API response with proper error checking."""
    if response.status_code != 200:
        print(f"ReMe API Error: {response.status_code}")
        print(response.text)
        return None
    return response.json()


def delete_workspace(workspace_id: str, api_url: str = "http://0.0.0.0:8002/"):
    """Delete the current workspace from the vector store."""
    response = requests.post(
        url=f"{api_url}vector_store",
        json={
            "workspace_id": workspace_id,
            "action": "delete",
        },
    )
    result = handle_api_response(response)
    if result:
        print(f"Workspace '{workspace_id}' deleted successfully")


def dump_memory(workspace_id: str, path: str = "./", api_url: str = "http://0.0.0.0:8002/"):
    """Dump the vector store memories to disk."""
    response = requests.post(
        url=f"{api_url}vector_store",
        json={
            "workspace_id": workspace_id,
            "action": "dump",
            "path": path,
        },
    )
    result = handle_api_response(response)
    if result:
        print(f"Memory dumped to {path}")


def load_memory(workspace_id: str, path: str = "docs/library", api_url: str = "http://0.0.0.0:8002/"):
    """Load memories from disk into the vector store."""
    response = requests.post(
        url=f"{api_url}vector_store",
        json={
            "workspace_id": workspace_id,
            "action": "load",
            "path": path,
        },
    )
    result = handle_api_response(response)
    if result:
        print(f"Memory loaded from {path}")


def count_memories(workspace_id: str, api_url: str = "http://0.0.0.0:8002/") -> int | None:
    """Count the number of memories in the vector store workspace."""
    response = requests.post(
        url=f"{api_url}vector_store",
        json={
            "workspace_id": workspace_id,
            "action": "count",
        },
    )
    result = handle_api_response(response)
    if not result:
        return None
    action_result = (result.get("metadata", {}) or {}).get("action_result", None)
    try:
        return int(action_result)
    except Exception:
        return None


def get_all_scores_from_session(session_dir: str) -> tuple[float, float, float]:
    """Get Detection score, Localization accuracy, and RCA accuracy by comparing submission with ground truth."""
    try:
        gt_path = os.path.join(session_dir, "ground_truth.json")
        sub_path = os.path.join(session_dir, "submission.json")

        if not os.path.exists(gt_path) or not os.path.exists(sub_path):
            return 0.0, 0.0, 0.0

        with open(gt_path) as f:
            gt = json.load(f)
        with open(sub_path) as f:
            submission = json.load(f)

        # Detection Score
        detection_score = 0.0
        try:
            parsed_detect_sub = DetectionSubmission.model_validate({"is_anomaly": submission.get("is_anomaly", False)})
            if gt["is_anomaly"] == parsed_detect_sub.is_anomaly:
                detection_score = 1.0
        except Exception:
            detection_score = 0.0

        # Localization accuracy
        loc_acc = 0.0
        try:
            loc_acc, _, _, _ = LocalizationTask().eval(
                submission={"faulty_devices": submission.get("faulty_devices", [])},
                gt={"faulty_devices": gt.get("faulty_devices", [])},
            )
            if loc_acc == -1.0:
                loc_acc = 0.0
        except Exception:
            loc_acc = 0.0

        # RCA accuracy
        rca_acc = 0.0
        try:
            rca_acc, _, _, _ = RCATask().eval(
                submission={"root_cause_name": submission.get("root_cause_name", [])},
                gt={"root_cause_name": gt.get("root_cause_name", [])},
            )
            if rca_acc == -1.0:
                rca_acc = 0.0
        except Exception:
            rca_acc = 0.0

        return detection_score, loc_acc, rca_acc
    except Exception:
        return 0.0, 0.0, 0.0


def get_llm_judge_score_from_session(session_dir: str) -> int:
    """Get LLM Judge overall_score from llm_judge.json."""
    try:
        judge_path = os.path.join(session_dir, "llm_judge.json")
        if not os.path.exists(judge_path):
            return 0
        with open(judge_path) as f:
            judge_result = json.load(f)
        return judge_result.get("scores", {}).get("overall_score", {}).get("score", 0)
    except Exception:
        return 0


def run_benchmark(
    backend_model: str = "gpt-5-mini",
    max_steps: int = 40,
    judge_model: str = "qwen3:32b",
    agent_type: str = "react",
    mode: str = "online",
    use_memory: bool = False,
    use_memory_addition: bool = False,
    use_memory_deletion: bool = False,
    freq_threshold: int = 5,
    utility_threshold: float = 0.5,
    memory_base_url: str = "http://0.0.0.0:8002/",
    memory_workspace_id: str = "nika_v1",
    benchmark_file_name: str = "benchmark_selected.csv",
    num_trials: int = 1,
    experiment_name: str | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    num_samples: int = 8,
    experience_pool_dir: str | None = None,
):
    """Run benchmark tests based on the benchmark.csv file.

    Args:
        num_trials: Number of retry attempts per task. If > 1, failed attempts
                   generate memories that are used in subsequent retries.
        experiment_name: Optional name to organize results under a separate directory.
                        Results will be saved to results/<experiment_name>/<problem>/...
        seed: Optional seed for reproducible fault injection. Use the same seed 
              across experiments (e.g., with/without memory) for fair comparison.
    """
    # Set experiment name for session directory organization
    set_experiment_name(experiment_name)

    build_offline_pool = mode == "offline"
    if build_offline_pool:
        use_memory = True
        use_memory_addition = True
        use_memory_deletion = False
    
    if experience_pool_dir is None:
        experience_pool_dir = DEFAULT_EXPERIENCE_POOL_DIR

    benchmark_file = os.path.join(cur_dir, benchmark_file_name)
    df = pl.read_csv(benchmark_file)
    case_log_path = init_benchmark_case_log(experiment_name)

    print(f"Running benchmark with memory={'enabled' if use_memory else 'disabled'}")
    if build_offline_pool:
        print(f"Backend model: {backend_model}, Max steps: {max_steps}, Num samples: {num_samples}")
    else:
        print(f"Backend model: {backend_model}, Max steps: {max_steps}, Num trials: {num_trials}")
    if experiment_name:
        print(f"Experiment name: {experiment_name} (results will be saved to results/{experiment_name}/...)")
    if seed is not None:
        print(f"Using fixed seed: {seed} (for reproducible fault injection)")
    if build_offline_pool:
        print("Offline pool build mode: memory retrieval disabled, failure experiences retained")
        print(f"Experience pool dir: {experience_pool_dir}")
        print(f"Sampling temperature: {temperature}")

    total_tasks = len(df)
    pbar = tqdm(df.iter_rows(named=True), total=total_tasks, desc="Benchmark Progress")
    for row in pbar:
        problem = row["problem"]
        scenario = row["scenario"]
        topo_size = row["topo_size"]

        pbar.set_description(f"[{problem}|{scenario}|{topo_size}]")

        case_status = "completed"
        error_stage = ""
        error_type = ""
        error_message = ""
        completed_trials = 0
        total_trials = num_samples if build_offline_pool else num_trials
        stage = ""
        last_eval_result = None
        last_eval_result_recorded = False

        # Track previous memories for retry logic
        previous_memories: list = []
        final_detection_score = 0.0
        final_loc_acc = 0.0
        final_rca_acc = 0.0
        final_llm_judge_score = 0.0
        sampled_trajectories: list[dict] = []

        try:
            enable_retry = (not build_offline_pool) and num_trials > 1

            # Run multiple trials if num_trials > 1, or N samples in offline pool mode
            for trial_id in range(total_trials):
                pbar.set_postfix(trial=f"{trial_id + 1}/{total_trials}")

                # Step 0: Wipe Kathara to ensure clean state (prevents contamination from previous runs)
                stage = "wipe_kathara"
                wipe_kathara()

                # Step 1: Start Network Environment (redeploy for each trial to reset state)
                stage = "start_net_env"
                start_net_env(scenario, topo_size=topo_size, redeploy=True, experiment_name=experiment_name)

                # Step 2: Inject Failure
                stage = "inject_failure"
                inject_failure(problem_names=[problem], seed=seed)

                # Step 3: Start Agent with previous_memories from failed attempts
                stage = "start_agent"
                agent = start_agent(
                    agent_type=agent_type,
                    backend_model=backend_model,
                    max_steps=max_steps,
                    use_memory=use_memory,
                    use_memory_addition=use_memory_addition,
                    use_memory_deletion=use_memory_deletion,
                    freq_threshold=freq_threshold,
                    utility_threshold=utility_threshold,
                    memory_base_url=memory_base_url,
                    memory_workspace_id=memory_workspace_id,
                    previous_memories=previous_memories if (enable_retry and trial_id > 0) else None,
                    temperature=temperature,
                    use_memory_retrieval=use_memory and (not build_offline_pool),
                    enable_memory_store=use_memory_addition or build_offline_pool,
                )

                # Step 4: Evaluate Results
                stage = "eval_results"
                last_eval_result_recorded = False
                last_eval_result = eval_results(judge_model=judge_model, destroy_env=False, record_summary=False)

                # Step 5: Get score and handle memory operations
                stage = "score_results"
                detection_score = 0.0
                loc_acc = 0.0
                rca_acc = 0.0
                llm_judge_score = 0.0
                is_perfect = False

                if agent:
                    if experiment_name:
                        session_dir = os.path.join(RESULTS_DIR, experiment_name, problem, scenario)
                    else:
                        session_dir = os.path.join(RESULTS_DIR, problem, scenario)
                    if os.path.exists(session_dir):
                        session_dirs = [
                            os.path.join(session_dir, d)
                            for d in os.listdir(session_dir)
                            if os.path.isdir(os.path.join(session_dir, d))
                        ]
                        if topo_size:
                            topo_prefix = f"{topo_size}-"
                            session_dirs = [
                                candidate_dir
                                for candidate_dir in session_dirs
                                if os.path.basename(candidate_dir).startswith(topo_prefix)
                            ]
                        if session_dirs:
                            latest_session_dir = str(max(session_dirs, key=os.path.getmtime))
                            detection_score, loc_acc, rca_acc = get_all_scores_from_session(latest_session_dir)
                            llm_judge_score = get_llm_judge_score_from_session(latest_session_dir)
                            print(
                                "  Scores - Detection: "
                                f"{detection_score}, Loc Acc: {loc_acc}, RCA Acc: {rca_acc}, LLM Judge: {llm_judge_score}"
                            )

                            # Perfect score condition: LLM Judge overall_score >= 4
                            # is_perfect = llm_judge_score >= 4
                            # Perfect score condition: Detection score == 1.0 and Loc Acc == 1.0 and RCA Acc == 1.0
                            is_perfect = detection_score == 1.0 and loc_acc == 1.0 and rca_acc == 1.0

                            # Memory addition logic (for num_trials retry)
                            if use_memory and use_memory_addition and not build_offline_pool:
                                pool_count_before = count_memories(memory_workspace_id, api_url=memory_base_url)
                                # Pass 1.0 if perfect, else 0.0 (simplification for memory utility)
                                combined_score = 1.0 if is_perfect else 0.0
                                task_history = getattr(agent, "task_history", []) or []

                                new_memories, memory_extraction_info = agent.store_memory_from_result(
                                    task_id=problem,
                                    task_history=task_history,
                                    score=combined_score,
                                    keep_failure_memories=build_offline_pool,
                                    rewrite_on_failure=not build_offline_pool,
                                )

                                pool_count_after_add = count_memories(memory_workspace_id, api_url=memory_base_url)
                                memory_extraction_info["experience_pool_count_before"] = pool_count_before
                                memory_extraction_info["experience_pool_count_after_add"] = pool_count_after_add
                                if pool_count_before is not None and pool_count_after_add is not None:
                                    memory_extraction_info["experience_pool_delta_add"] = (
                                        pool_count_after_add - pool_count_before
                                    )

                                # Save memory extraction info to session directory
                                memory_extraction_path = os.path.join(latest_session_dir, "memory_extraction.json")
                                with open(memory_extraction_path, "w") as f:
                                    json.dump(memory_extraction_info, f, indent=2, ensure_ascii=False, default=str)
                                print(f"  Memory extraction info saved to {memory_extraction_path}")
                                if pool_count_before is not None and pool_count_after_add is not None:
                                    print(
                                        "  Experience pool (add): "
                                        f"before={pool_count_before}, after={pool_count_after_add}, "
                                        f"delta={pool_count_after_add - pool_count_before}"
                                    )

                                if enable_retry:
                                    if not is_perfect and new_memories:
                                        # Failed: use rewritten context for next retry when available
                                        rewrite_info = memory_extraction_info.get("rewrite", {})
                                        rewritten_context = rewrite_info.get("rewritten_context", "")
                                        if rewritten_context:
                                            previous_memories = [
                                                {
                                                    # "when_to_use": "Retry the same task after failure reflection",
                                                    "content": rewritten_context,
                                                }
                                            ]
                                        else:
                                            previous_memories = new_memories
                                    else:
                                        # Success or no memories: clear previous_memories
                                        previous_memories = []
                            elif build_offline_pool and agent:
                                combined_score = 1.0 if is_perfect else 0.0
                                task_history = getattr(agent, "task_history", []) or []
                                trajectory = agent.get_traj_from_messages(
                                    task_id=problem,
                                    messages=task_history,
                                    score=combined_score,
                                )
                                sampled_trajectories.append(trajectory)

                            # Update memory usage information
                            if use_memory and agent.retrieved_memory_list:
                                agent.update_memory_information(
                                    agent.retrieved_memory_list,
                                    update_utility=is_perfect,
                                )

                            # Delete low-quality memories if enabled
                            if use_memory and use_memory_deletion:
                                agent.delete_memory()

                            # Record evaluation summary row (with memory pool stats when available)
                            if last_eval_result and not last_eval_result_recorded:
                                added_count = 0
                                if use_memory and use_memory_addition and not build_offline_pool:
                                    action = memory_extraction_info.get("action", "")
                                    if action != "deleted_for_retry":
                                        added_count = len(new_memories)
                                pool_total_count = (
                                    count_memories(memory_workspace_id, api_url=memory_base_url) if use_memory else None
                                )
                                last_eval_result.experience_added_count = added_count
                                last_eval_result.experience_pool_total_count = pool_total_count
                                record_eval_result(last_eval_result)
                                last_eval_result_recorded = True

                if last_eval_result and not last_eval_result_recorded:
                    pool_total_count = count_memories(memory_workspace_id, api_url=memory_base_url) if use_memory else None
                    last_eval_result.experience_added_count = 0
                    last_eval_result.experience_pool_total_count = pool_total_count
                    record_eval_result(last_eval_result)
                    last_eval_result_recorded = True

                final_detection_score = detection_score
                final_loc_acc = loc_acc
                final_rca_acc = rca_acc
                final_llm_judge_score = llm_judge_score
                completed_trials = trial_id + 1

                # Success: stop retrying
                if is_perfect and not build_offline_pool:
                    print(f"  Task succeeded on trial {trial_id + 1}")
                    break

            if build_offline_pool and sampled_trajectories and agent:
                print(f"  Building experience pool from {len(sampled_trajectories)} trajectories")
                pool_count_before = count_memories(memory_workspace_id, api_url=memory_base_url)
                new_memories, metadata = agent.add_memory(sampled_trajectories, rewrite=False)
                pool_count_after = count_memories(memory_workspace_id, api_url=memory_base_url)
                metadata = dict(metadata or {})
                metadata["experience_pool_count_before"] = pool_count_before
                metadata["experience_pool_count_after"] = pool_count_after
                if pool_count_before is not None and pool_count_after is not None:
                    metadata["experience_pool_delta"] = pool_count_after - pool_count_before
                if new_memories:
                    print(f"  Added {len(new_memories)} memories to pool")
                if pool_count_before is not None and pool_count_after is not None:
                    print(
                        "  Experience pool (offline build): "
                        f"before={pool_count_before}, after={pool_count_after}, "
                        f"delta={pool_count_after - pool_count_before}"
                    )
                if experiment_name:
                    pool_log_dir = os.path.join(RESULTS_DIR, experiment_name, problem, scenario)
                else:
                    pool_log_dir = os.path.join(RESULTS_DIR, problem, scenario)
                os.makedirs(pool_log_dir, exist_ok=True)
                pool_metadata_path = os.path.join(pool_log_dir, "memory_extraction_batch.json")
                with open(pool_metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            if last_eval_result and not last_eval_result_recorded:
                try:
                    record_eval_result(last_eval_result)
                    last_eval_result_recorded = True
                except Exception:
                    pass
            case_status = "failed"
            error_stage = stage or "unknown"
            error_type = type(e).__name__
            error_message = str(e)
            print(f"  Case failed at {error_stage}: {error_type}: {error_message}")
        finally:
            if case_status == "completed":
                print(
                    "  Final Scores - Detection: "
                    f"{final_detection_score}, Loc Acc: {final_loc_acc}, "
                    f"RCA Acc: {final_rca_acc}, LLM Judge: {final_llm_judge_score}"
                )

                # Dump memory after each task to prevent data loss on interruption
                if use_memory:
                    if build_offline_pool:
                        dump_path = os.path.join(experience_pool_dir, memory_workspace_id)
                        os.makedirs(dump_path, exist_ok=True)
                    elif experiment_name:
                        dump_path = f"{RESULTS_DIR}/{experiment_name}/memory_dump_{memory_workspace_id}"
                    else:
                        dump_path = f"{RESULTS_DIR}/memory_dump_{memory_workspace_id}"
                    dump_memory(
                        workspace_id=memory_workspace_id,
                        path=dump_path,
                        api_url=memory_base_url,
                    )

            # Finally, destroy the network environment
            try:
                net_env = get_net_env_instance(scenario, topo_size=topo_size)
                if net_env.lab_exists():
                    net_env.undeploy()
            except Exception as cleanup_err:
                print(f"Warning: cleanup failed: {cleanup_err}")

            append_benchmark_case_log(
                case_log_path,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "problem": problem,
                    "scenario": scenario,
                    "topo_size": topo_size,
                    "status": case_status,
                    "completed_trials": completed_trials,
                    "num_trials": total_trials,
                    "error_stage": error_stage,
                    "error_type": error_type,
                    "error_message": error_message,
                    "detection_score": final_detection_score,
                    "loc_acc": final_loc_acc,
                    "rca_acc": final_rca_acc,
                    "llm_judge_score": final_llm_judge_score,
                },
            )


def main():
    parser = argparse.ArgumentParser(description="Run NIKA benchmark with optional ReMe memory")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["online", "offline"],
        default="online",
        help="Run mode: online (benchmark) or offline (experience pool build).",
    )
    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="Enable ReMe memory system (retrieval + update)",
    )
    parser.add_argument(
        "--use-memory-addition",
        action="store_true",
        help="Enable memory addition after successful tasks",
    )
    parser.add_argument(
        "--use-memory-deletion",
        action="store_true",
        help="Enable low-quality memory deletion",
    )
    parser.add_argument(
        "--freq-threshold",
        type=int,
        default=5,
        help="Frequency threshold for memory deletion (default: 5)",
    )
    parser.add_argument(
        "--utility-threshold",
        type=float,
        default=0.5,
        help="Utility threshold for memory deletion (default: 0.5)",
    )
    parser.add_argument(
        "--backend-model",
        type=str,
        default="azure/gpt-5.2",
        help="Backend model for the agent (default: azure/gpt-5.2)",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="react",
        help="Agent type to run (default: react).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="Maximum steps for the agent (default: 40)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="azure/gpt-5.2",
        help="LLM model used for judgment (default: azure/gpt-5.2)",
    )
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default="benchmark_selected.csv",
        help="Benchmark CSV file name (default: benchmark_selected.csv)",
    )
    parser.add_argument(
        "--memory-workspace-id",
        type=str,
        default="nika_v1",
        help="ReMe workspace ID (default: nika_v1)",
    )
    parser.add_argument(
        "--memory-api-url",
        type=str,
        default="http://0.0.0.0:8002/",
        help="ReMe API URL (default: http://0.0.0.0:8002/)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of retry attempts per task (default: 1). Failed attempts generate memories for retries.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples per task when building offline experience pool (default: 8).",
    )
    parser.add_argument(
        "--experience-pool-dir",
        type=str,
        default=None,
        help="Directory for offline experience pool dumps (default: cookbook/nika/experience_pool).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature override (default: 0 for online; 0.9 for offline).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name to organize results. Results will be saved to results/<experiment_name>/<problem>/...",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional seed for reproducible fault injection. Use the same seed across experiments (e.g., with/without memory) for fair comparison.",
    )
    parser.add_argument(
        "--resume-memory",
        action="store_true",
        help="Load existing memory dump before starting (for resuming interrupted experiments).",
    )
    parser.add_argument(
        "--memory-dump-path",
        type=str,
        default=None,
        help=(
            "Optional path to an existing memory dump directory to load before running. "
            "If provided, it overrides the default resume-memory path derived from experiment-name."
        ),
    )
    args = parser.parse_args()
    agent_type = args.agent_type

    # Resolve mode and memory flags
    is_offline = args.mode == "offline"
    memory_requested = (
        args.use_memory
        or args.use_memory_addition
        or args.use_memory_deletion
        or args.resume_memory
        or args.memory_dump_path
        or is_offline
    )

    if is_offline:
        # Offline phase: build initial pool; retrieval and deletion are off.
        use_memory = True
        use_memory_addition = True
        use_memory_deletion = False
    else:
        # Online phase: always enable retrieval/add/delete.
        use_memory = True
        use_memory_addition = True
        use_memory_deletion = True

    experience_pool_dir = args.experience_pool_dir or DEFAULT_EXPERIENCE_POOL_DIR
    temperature = args.temperature
    if temperature is None:
        temperature = 0.9 if is_offline else 0.0

    # Initialize memory workspace if memory is enabled
    if use_memory:
        memory_dump_path = args.memory_dump_path
        if is_offline:
            dump_path = os.path.join(experience_pool_dir, args.memory_workspace_id)
            os.makedirs(dump_path, exist_ok=True)
            if args.resume_memory:
                load_path = memory_dump_path or dump_path
                print(f"Resuming from existing experience pool: {load_path}")
                load_memory(workspace_id=args.memory_workspace_id, path=load_path, api_url=args.memory_api_url)
            else:
                print("Initializing ReMe memory workspace for offline pool (fresh start)...")
                delete_workspace(workspace_id=args.memory_workspace_id, api_url=args.memory_api_url)
                time.sleep(2)
        else:
            if args.experiment_name:
                dump_path = f"{RESULTS_DIR}/{args.experiment_name}/memory_dump_{args.memory_workspace_id}"
            else:
                dump_path = f"{RESULTS_DIR}/memory_dump_{args.memory_workspace_id}"
            load_path = None
            if memory_dump_path:
                load_path = memory_dump_path
            elif args.resume_memory:
                load_path = dump_path

            if load_path:
                # Resume from existing memory dump
                print(f"Resuming from existing memory dump: {load_path}")
                load_memory(workspace_id=args.memory_workspace_id, path=load_path, api_url=args.memory_api_url)
            elif memory_requested:
                # Fresh start: delete existing workspace
                print("Initializing ReMe memory workspace (fresh start)...")
                delete_workspace(workspace_id=args.memory_workspace_id, api_url=args.memory_api_url)
                time.sleep(2)
            else:
                print("Using existing ReMe memory workspace (no reload).")

    # Run benchmark
    run_benchmark(
        backend_model=args.backend_model,
        max_steps=args.max_steps,
        judge_model=args.judge_model,
        agent_type=agent_type,
        use_memory=use_memory,
        use_memory_addition=use_memory_addition,
        use_memory_deletion=use_memory_deletion,
        freq_threshold=args.freq_threshold,
        utility_threshold=args.utility_threshold,
        memory_base_url=args.memory_api_url,
        memory_workspace_id=args.memory_workspace_id,
        benchmark_file_name=args.benchmark_file,
        num_trials=args.num_trials,
        experiment_name=args.experiment_name,
        seed=args.seed,
        temperature=temperature,
        mode=args.mode,
        num_samples=args.num_samples,
        experience_pool_dir=experience_pool_dir,
    )

    # Optionally dump memories after benchmark (final dump)
    if use_memory:
        if is_offline:
            dump_path = os.path.join(experience_pool_dir, args.memory_workspace_id)
            os.makedirs(dump_path, exist_ok=True)
        elif args.experiment_name:
            dump_path = f"{RESULTS_DIR}/{args.experiment_name}/memory_dump_{args.memory_workspace_id}"
        else:
            dump_path = f"{RESULTS_DIR}/memory_dump_{args.memory_workspace_id}"
        dump_memory(
            workspace_id=args.memory_workspace_id,
            path=dump_path,
            api_url=args.memory_api_url,
        )


if __name__ == "__main__":
    main()
