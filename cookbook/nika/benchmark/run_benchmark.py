import argparse
import json
import os
import time

import polars as pl
import requests
from tqdm import tqdm

from nika.config import RESULTS_DIR
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


def rewrite_failure_trajectory(
    workspace_id: str,
    trajectory: dict,
    query: str | None,
    api_url: str = "http://0.0.0.0:8002/",
) -> dict | None:
    """Rewrite failure memories using the summary_task_memory_rewrite flow."""
    resolved_query = query
    if not resolved_query:
        resolved_query = trajectory.get("metadata", {}).get("query", "")
    if not resolved_query:
        messages = trajectory.get("messages", [])
        if messages:
            resolved_query = messages[0].get("content", "")
    if not resolved_query:
        return None

    response = requests.post(
        url=f"{api_url}summary_task_memory_rewrite",
        json={
            "workspace_id": workspace_id,
            "trajectories": [trajectory],
            "query": resolved_query,
        },
    )
    return handle_api_response(response)


def get_all_scores_from_session(session_dir: str) -> tuple[float, float, float]:
    """Get Detection score, Localization F1, and RCA F1 by comparing submission with ground truth."""
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

        # Localization F1
        loc_f1 = 0.0
        try:
            _, _, _, loc_f1 = LocalizationTask().eval(
                submission={"faulty_devices": submission.get("faulty_devices", [])},
                gt={"faulty_devices": gt.get("faulty_devices", [])},
            )
            if loc_f1 == -1.0:
                loc_f1 = 0.0
        except Exception:
            loc_f1 = 0.0

        # RCA F1
        rca_f1 = 0.0
        try:
            _, _, _, rca_f1 = RCATask().eval(
                submission={"root_cause_name": submission.get("root_cause_name", [])},
                gt={"root_cause_name": gt.get("root_cause_name", [])},
            )
            if rca_f1 == -1.0:
                rca_f1 = 0.0
        except Exception:
            rca_f1 = 0.0

        return detection_score, loc_f1, rca_f1
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
    
    benchmark_file = os.path.join(cur_dir, benchmark_file_name)
    df = pl.read_csv(benchmark_file)

    print(f"Running benchmark with memory={'enabled' if use_memory else 'disabled'}")
    print(f"Backend model: {backend_model}, Max steps: {max_steps}, Num trials: {num_trials}")
    if experiment_name:
        print(f"Experiment name: {experiment_name} (results will be saved to results/{experiment_name}/...)")
    if seed is not None:
        print(f"Using fixed seed: {seed} (for reproducible fault injection)")

    total_tasks = len(df)
    pbar = tqdm(df.iter_rows(named=True), total=total_tasks, desc="Benchmark Progress")
    for row in pbar:
        problem = row["problem"]
        scenario = row["scenario"]
        topo_size = row["topo_size"]

        pbar.set_description(f"[{problem}|{scenario}|{topo_size}]")

        # Track previous memories for retry logic
        previous_memories: list = []
        final_detection_score = 0.0
        final_loc_f1 = 0.0
        final_rca_f1 = 0.0

        # Run multiple trials if num_trials > 1
        for trial_id in range(num_trials):
            pbar.set_postfix(trial=f"{trial_id + 1}/{num_trials}")

            # Step 0: Wipe Kathara to ensure clean state (prevents contamination from previous runs)
            wipe_kathara()

            # Step 1: Start Network Environment (redeploy for each trial to reset state)
            start_net_env(scenario, topo_size=topo_size, redeploy=True, experiment_name=experiment_name)

            # Step 2: Inject Failure
            inject_failure(problem_names=[problem], seed=seed)

            # Step 3: Start Agent with previous_memories from failed attempts
            agent = start_agent(
                agent_type="react",
                backend_model=backend_model,
                max_steps=max_steps,
                use_memory=use_memory,
                use_memory_addition=use_memory_addition,
                use_memory_deletion=use_memory_deletion,
                freq_threshold=freq_threshold,
                utility_threshold=utility_threshold,
                memory_base_url=memory_base_url,
                memory_workspace_id=memory_workspace_id,
                previous_memories=previous_memories if trial_id > 0 else None,
            )

            # Step 4: Evaluate Results
            eval_results(judge_model=judge_model, destroy_env=False)

            # Step 5: Get score and handle memory operations
            detection_score = 0.0
            loc_f1 = 0.0
            rca_f1 = 0.0
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
                        detection_score, loc_f1, rca_f1 = get_all_scores_from_session(latest_session_dir)
                        llm_judge_score = get_llm_judge_score_from_session(latest_session_dir)
                        print(f"  Scores - Detection: {detection_score}, Loc F1: {loc_f1}, RCA F1: {rca_f1}, LLM Judge: {llm_judge_score}")

                        # Perfect score condition: LLM Judge overall_score >= 4
                        is_perfect = llm_judge_score >= 4

                        # Memory addition logic (for num_trials retry)
                        if use_memory and use_memory_addition:
                            # Pass 1.0 if perfect, else 0.0 (simplification for memory utility)
                            combined_score = 1.0 if is_perfect else 0.0
                            task_history = getattr(agent, "task_history", []) or []
                            
                            new_memories, memory_extraction_info = agent.store_memory_from_result(
                                task_id=problem,
                                task_history=task_history,
                                score=combined_score,
                            )

                            rewritten_context = ""
                            rewrite_metadata = {}
                            if not is_perfect and new_memories:
                                trajectory = memory_extraction_info.get("trajectory")
                                query = None
                                if isinstance(trajectory, dict):
                                    query = trajectory.get("metadata", {}).get("query")
                                if trajectory:
                                    rewrite_result = rewrite_failure_trajectory(
                                        workspace_id=memory_workspace_id,
                                        trajectory=trajectory,
                                        query=query,
                                        api_url=memory_base_url,
                                    )
                                    if rewrite_result:
                                        rewritten_context = rewrite_result.get("answer", "")
                                        rewrite_metadata = rewrite_result.get("metadata", {})
                            if rewritten_context:
                                memory_extraction_info["rewrite"] = {
                                    "rewritten_context": rewritten_context,
                                    "metadata": rewrite_metadata,
                                }

                            # Save memory extraction info to session directory
                            memory_extraction_path = os.path.join(latest_session_dir, "memory_extraction.json")
                            with open(memory_extraction_path, "w") as f:
                                json.dump(memory_extraction_info, f, indent=2, ensure_ascii=False, default=str)
                            print(f"  Memory extraction info saved to {memory_extraction_path}")

                            if not is_perfect and new_memories:
                                # Failed: use rewritten context for next retry when available
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

                        # Update memory usage information
                        if use_memory and agent.retrieved_memory_list:
                            agent.update_memory_information(
                                agent.retrieved_memory_list,
                                update_utility=is_perfect,
                            )

                        # Delete low-quality memories if enabled
                        if use_memory and use_memory_deletion:
                            agent.delete_memory()

            final_detection_score = detection_score
            final_loc_f1 = loc_f1
            final_rca_f1 = rca_f1
            final_llm_judge_score = llm_judge_score

            # Success: stop retrying
            if is_perfect:
                print(f"  Task succeeded on trial {trial_id + 1}")
                break

        print(f"  Final Scores - Detection: {final_detection_score}, Loc F1: {final_loc_f1}, RCA F1: {final_rca_f1}, LLM Judge: {final_llm_judge_score}")

        # Dump memory after each task to prevent data loss on interruption
        if use_memory:
            if experiment_name:
                dump_path = f"{RESULTS_DIR}/{experiment_name}/memory_dump_{memory_workspace_id}"
            else:
                dump_path = f"{RESULTS_DIR}/memory_dump_{memory_workspace_id}"
            dump_memory(
                workspace_id=memory_workspace_id,
                path=dump_path,
                api_url=memory_base_url,
            )

        # Finally, destroy the network environment
        net_env = get_net_env_instance(scenario, topo_size=topo_size)
        if net_env.lab_exists():
            net_env.undeploy()


def main():
    parser = argparse.ArgumentParser(description="Run NIKA benchmark with optional ReMe memory")
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
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name to organize results. Results will be saved to results/<experiment_name>/<problem>/...",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible fault injection. Use the same seed across experiments (e.g., with/without memory) for fair comparison.",
    )
    parser.add_argument(
        "--resume-memory",
        action="store_true",
        help="Load existing memory dump before starting (for resuming interrupted experiments).",
    )
    args = parser.parse_args()

    # If use-memory is set, automatically enable addition and deletion
    use_memory_addition = args.use_memory_addition or args.use_memory
    use_memory_deletion = args.use_memory_deletion or args.use_memory

    # Initialize memory workspace if memory is enabled
    if args.use_memory:
        if args.experiment_name:
            dump_path = f"{RESULTS_DIR}/{args.experiment_name}/memory_dump_{args.memory_workspace_id}"
        else:
            dump_path = f"{RESULTS_DIR}/memory_dump_{args.memory_workspace_id}"
        
        if args.resume_memory:
            # Resume from existing memory dump
            print(f"Resuming from existing memory dump: {dump_path}")
            load_memory(workspace_id=args.memory_workspace_id, path=dump_path, api_url=args.memory_api_url)
        else:
            # Fresh start: delete existing workspace
            print("Initializing ReMe memory workspace (fresh start)...")
        delete_workspace(workspace_id=args.memory_workspace_id, api_url=args.memory_api_url)
        time.sleep(2)

    # Run benchmark
    run_benchmark(
        backend_model=args.backend_model,
        max_steps=args.max_steps,
        judge_model=args.judge_model,
        use_memory=args.use_memory,
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
    )

    # Optionally dump memories after benchmark (final dump)
    if args.use_memory:
        if args.experiment_name:
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
