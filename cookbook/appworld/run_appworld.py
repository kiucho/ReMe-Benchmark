# flake8: noqa: E402
import argparse
import os
import time

import ray
import requests
from ray import logger

os.environ["APPWORLD_ROOT"] = "."
from dotenv import load_dotenv

load_dotenv("../../.env")

import json
from pathlib import Path

from appworld import load_task_ids

from appworld_react_agent import AppworldReactAgent


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AppWorld ReAct agent benchmark")

    parser.add_argument(
        "--mode",
        default="w_mem_warm",
        choices=["wo_mem", "w_mem_cold", "w_mem_warm", "w_mem_offline"],
        help=(
            "Benchmark mode: wo_mem (no memory), "
            "w_mem_cold (fresh empty workspace; accumulate online), "
            "w_mem_warm (load provided starting memory, then evaluate and keep accumulating), "
            "w_mem_offline (build offline memory pool with sampling)."
        ),
    )

    parser.add_argument(
        "--backend-model", default="gpt-oss-120b", help="LLM backend model name"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature override. Default: 0.9 for w_mem_offline, 0.0 otherwise.",
    )
    parser.add_argument(
        "--dataset-name", default="test_normal", help="AppWorld dataset name"
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Output experiment name (jsonl file stem). Default: derived from dataset/mode.",
    )

    # Memory benchmarks must run sequentially to avoid cross-task interference.
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=4)

    parser.add_argument("--use-memory-addition", action="store_true", default=True)
    parser.add_argument(
        "--no-use-memory-addition", action="store_false", dest="use_memory_addition"
    )
    parser.add_argument("--use-memory-deletion", action="store_true", default=True)
    parser.add_argument(
        "--no-use-memory-deletion", action="store_false", dest="use_memory_deletion"
    )

    parser.add_argument("--rewrite-on-failure", action="store_true", default=True)
    parser.add_argument(
        "--no-rewrite-on-failure", action="store_false", dest="rewrite_on_failure"
    )

    parser.add_argument("--delete-freq", type=int, default=5)
    parser.add_argument("--freq-threshold", type=int, default=5)
    parser.add_argument("--utility-threshold", type=float, default=0.5)

    parser.add_argument("--memory-workspace-id", default="appworld")
    parser.add_argument("--memory-api-url", default="http://0.0.0.0:8002/")
    parser.add_argument(
        "--starting-memory-path",
        default=None,
        help=(
            "Path to a prebuilt memory dump to load as starting memory (required for warm start)."
        ),
    )
    parser.add_argument(
        "--experience-pool-dir",
        default="./experience_pool",
        help="Directory for offline memory pool dumps.",
    )
    parser.add_argument(
        "--resume-memory",
        action="store_true",
        help="Resume offline pool build from an existing dump path.",
    )

    return parser


def handle_api_response(response: requests.Response):
    """Handle API response with proper error checking"""
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    return response.json()


def delete_workspace(workspace_id: str, api_url: str = "http://0.0.0.0:8002/"):
    """Delete the current workspace from the vector store"""
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


def dump_memory(
    workspace_id: str, path: str = "./", api_url: str = "http://0.0.0.0:8002/"
) -> bool:
    """Dump the vector store memories to disk"""
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
        return True
    return False


def load_memory(
    workspace_id: str, path: str = "docs/library", api_url: str = "http://0.0.0.0:8002/"
):
    """Load memories from disk into the vector store"""
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


def run_agent(
    model_name: str,
    dataset_name: str,
    experiment_name: str,
    max_workers: int,
    num_trials: int = 1,
    use_memory: bool = False,
    use_memory_addition: bool = False,
    use_memory_deletion: bool = False,
    rewrite_on_failure: bool = True,
    delete_freq: int = 10,
    freq_threshold: int = 5,
    utility_threshold: float = 0.5,
    workspace_id: str = "appworld_v1",
    api_url: str = "http://0.0.0.0:8002/",
    batch_size: int = 4,
    temperature: float = 0.0,
    use_memory_retrieval: bool = True,
    keep_failure_memories: bool = False,
    stop_on_success: bool = True,
):
    output_dir: Path = Path(f"./exp_result/{experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file_stem = model_name.replace("/", "__")
    output_file = output_dir / f"{model_file_stem}.jsonl"

    task_ids = load_task_ids(dataset_name)
    result: list = []

    def dump_file():
        with open(output_file, "a") as f:
            for x in result:
                f.write(json.dumps(x) + "\n")

    if max_workers > 1:
        # Process tasks in batches
        total_tasks = len(task_ids)
        num_batches = (total_tasks + batch_size - 1) // batch_size  # Ceiling division

        logger.info(
            f"Total tasks: {total_tasks}, Batch size: {batch_size}, Number of batches: {num_batches}"
        )

        for batch_idx in range(num_batches):
            # Initialize Ray for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_tasks)
            batch_task_ids = task_ids[start_idx:end_idx]

            logger.info(
                f"Starting batch {batch_idx + 1}/{num_batches} with {len(batch_task_ids)} tasks"
            )

            # Initialize Ray with the number of CPUs needed for this batch
            ray.init(num_cpus=len(batch_task_ids))

            RemoteAppworldReactAgent = ray.remote(AppworldReactAgent)

            future_list: list = []
            for i, task_id in enumerate(batch_task_ids):
                actor = RemoteAppworldReactAgent.remote(
                    index=start_idx + i,
                    model_name=model_name,
                    task_ids=[task_id],
                    experiment_name=experiment_name,
                    num_trials=num_trials,
                    use_memory=use_memory,
                    use_memory_addition=use_memory_addition,
                    use_memory_deletion=use_memory_deletion,
                    rewrite_on_failure=rewrite_on_failure,
                    delete_freq=delete_freq,
                    freq_threshold=freq_threshold,
                    utility_threshold=utility_threshold,
                    memory_workspace_id=workspace_id,
                    memory_base_url=api_url,
                    temperature=temperature,
                    use_memory_retrieval=use_memory_retrieval,
                    keep_failure_memories=keep_failure_memories,
                    stop_on_success=stop_on_success,
                )
                future = actor.execute.remote()
                future_list.append(future)
                time.sleep(1)

            logger.info(
                f"Batch {batch_idx + 1} submit complete, waiting for results..."
            )

            # Collect results from this batch
            for i, (task_id, future) in enumerate(zip(batch_task_ids, future_list)):
                try:
                    t_result = ray.get(future)
                    if t_result:
                        if isinstance(t_result, list):
                            result.extend(t_result)
                        else:
                            result.append(t_result)
                except Exception as e:
                    logger.exception(f"run ray error with task_id={task_id}")

                logger.info(
                    f"Batch {batch_idx + 1}: task {i + 1}/{len(batch_task_ids)} complete"
                )

            # Shutdown Ray to free resources before next batch
            ray.shutdown()
            logger.info(
                f"Batch {batch_idx + 1}/{num_batches} complete, Ray resources released"
            )

            # Optional: small delay between batches
            if batch_idx < num_batches - 1:
                time.sleep(2)

        dump_file()

    else:
        for index, task_id in enumerate(task_ids):
            agent = AppworldReactAgent(
                index=index,
                model_name=model_name,
                task_ids=[task_id],
                experiment_name=experiment_name,
                num_trials=num_trials,
                use_memory=use_memory,
                use_memory_addition=use_memory_addition,
                use_memory_deletion=use_memory_deletion,
                rewrite_on_failure=rewrite_on_failure,
                delete_freq=delete_freq,
                freq_threshold=freq_threshold,
                utility_threshold=utility_threshold,
                memory_workspace_id=workspace_id,
                memory_base_url=api_url,
                temperature=temperature,
                use_memory_retrieval=use_memory_retrieval,
                keep_failure_memories=keep_failure_memories,
                stop_on_success=stop_on_success,
            )
            task_results = agent.execute()
            if isinstance(task_results, list):
                result.extend(task_results)
            else:
                result.append(task_results)
        dump_file()


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.mode not in {"wo_mem", "w_mem_cold", "w_mem_warm", "w_mem_offline"}:
        raise ValueError(f"Unsupported mode: {args.mode}")

    experiment_name = args.experiment_name
    if not experiment_name:
        experiment_name = f"{args.dataset_name}_{args.mode}"

    resolved_temperature = args.temperature
    if resolved_temperature is None:
        resolved_temperature = 0.9 if args.mode == "w_mem_offline" else 0.0

    def reset_workspace(load_path: str | None = None):
        logger.info("Deleting workspace...")
        delete_workspace(
            workspace_id=args.memory_workspace_id, api_url=args.memory_api_url
        )
        time.sleep(2)
        if load_path:
            logger.info(f"Loading starting memories from {load_path}...")
            load_memory(
                workspace_id=args.memory_workspace_id,
                path=load_path,
                api_url=args.memory_api_url,
            )

    def write_memory_manifest(
        manifest_path: Path,
        *,
        starting_memory_path: str | None,
        final_dump_path: str,
        dump_succeeded: bool,
    ):
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "experiment_name": experiment_name,
            "mode": args.mode,
            "workspace_id": args.memory_workspace_id,
            "memory_api_url": args.memory_api_url,
            "num_runs": args.num_runs,
            "num_trials": args.num_trials,
            "num_samples": args.num_samples,
            "use_memory_addition": args.use_memory_addition,
            "use_memory_deletion": args.use_memory_deletion,
            "rewrite_on_failure": args.rewrite_on_failure,
            "starting_memory_path": starting_memory_path,
            "experience_pool_dir": args.experience_pool_dir,
            "resume_memory": args.resume_memory,
            "final_dump_path": final_dump_path,
            "final_dump_succeeded": dump_succeeded,
            "temperature": resolved_temperature,
            "dumped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info("Memory manifest saved to {}", manifest_path)

    def dump_final_memory_and_manifest(starting_memory_path: str | None = None):
        memory_dir = Path(f"./exp_result/{experiment_name}/memory")
        final_dump_dir = memory_dir / "final"
        final_dump_dir.mkdir(parents=True, exist_ok=True)

        final_dump_path = str(final_dump_dir.resolve())
        dump_succeeded = dump_memory(
            workspace_id=args.memory_workspace_id,
            path=final_dump_path,
            api_url=args.memory_api_url,
        )
        if not dump_succeeded:
            logger.warning(
                "Failed to dump final memory snapshot to {}",
                final_dump_path,
            )

        write_memory_manifest(
            memory_dir / "manifest.json",
            starting_memory_path=starting_memory_path,
            final_dump_path=final_dump_path,
            dump_succeeded=dump_succeeded,
        )

    # wo_mem: no memory calls
    if args.mode == "wo_mem":
        for _ in range(args.num_runs):
            run_agent(
                model_name=args.backend_model,
                dataset_name=args.dataset_name,
                experiment_name=experiment_name,
                max_workers=args.max_workers,
                num_trials=args.num_trials,
                use_memory=False,
                use_memory_addition=False,
                use_memory_deletion=False,
                rewrite_on_failure=args.rewrite_on_failure,
                delete_freq=args.delete_freq,
                freq_threshold=args.freq_threshold,
                utility_threshold=args.utility_threshold,
                workspace_id=args.memory_workspace_id,
                api_url=args.memory_api_url,
                batch_size=args.batch_size,
                temperature=resolved_temperature,
            )
        return

    # w_mem_cold: start empty, then accumulate sequentially
    if args.mode == "w_mem_cold":
        if args.starting_memory_path:
            logger.warning(
                "Ignoring --starting-memory-path for w_mem_cold (cold start builds from empty workspace)."
            )
        for _ in range(args.num_runs):
            reset_workspace(load_path=None)
            run_agent(
                model_name=args.backend_model,
                dataset_name=args.dataset_name,
                experiment_name=experiment_name,
                max_workers=args.max_workers,
                num_trials=args.num_trials,
                use_memory=True,
                use_memory_addition=args.use_memory_addition,
                use_memory_deletion=args.use_memory_deletion,
                rewrite_on_failure=args.rewrite_on_failure,
                delete_freq=args.delete_freq,
                freq_threshold=args.freq_threshold,
                utility_threshold=args.utility_threshold,
                workspace_id=args.memory_workspace_id,
                api_url=args.memory_api_url,
                batch_size=args.batch_size,
                temperature=resolved_temperature,
            )
        dump_final_memory_and_manifest(starting_memory_path=None)
        return

    if args.mode == "w_mem_offline":
        experience_pool_path = Path(args.experience_pool_dir) / args.memory_workspace_id
        experience_pool_path.mkdir(parents=True, exist_ok=True)
        resolved_pool_path = str(experience_pool_path.resolve())

        if args.resume_memory:
            resume_path = args.starting_memory_path or resolved_pool_path
            if Path(resume_path).exists():
                logger.info("Resuming offline pool from {}", resume_path)
                reset_workspace(load_path=resume_path)
            else:
                logger.warning(
                    "Resume path not found ({}). Starting offline pool from empty workspace.",
                    resume_path,
                )
                reset_workspace(load_path=None)
        else:
            reset_workspace(load_path=args.starting_memory_path)

        run_agent(
            model_name=args.backend_model,
            dataset_name=args.dataset_name,
            experiment_name=experiment_name,
            max_workers=args.max_workers,
            num_trials=args.num_samples,
            use_memory=True,
            use_memory_addition=True,
            use_memory_deletion=False,
            rewrite_on_failure=False,
            delete_freq=args.delete_freq,
            freq_threshold=args.freq_threshold,
            utility_threshold=args.utility_threshold,
            workspace_id=args.memory_workspace_id,
            api_url=args.memory_api_url,
            batch_size=args.batch_size,
            temperature=resolved_temperature,
            use_memory_retrieval=False,
            keep_failure_memories=True,
            stop_on_success=False,
        )

        dump_succeeded = dump_memory(
            workspace_id=args.memory_workspace_id,
            path=resolved_pool_path,
            api_url=args.memory_api_url,
        )
        if not dump_succeeded:
            logger.warning("Failed to dump offline pool to {}", resolved_pool_path)

        write_memory_manifest(
            Path(f"./exp_result/{experiment_name}/memory/manifest.json"),
            starting_memory_path=args.starting_memory_path,
            final_dump_path=resolved_pool_path,
            dump_succeeded=dump_succeeded,
        )
        return

    # w_mem_warm: load provided starting memory, then evaluate and keep accumulating.
    starting_memory_path = args.starting_memory_path
    if not starting_memory_path:
        raise ValueError("w_mem_warm requires --starting-memory-path")

    for _ in range(args.num_runs):
        reset_workspace(load_path=starting_memory_path)
        run_agent(
            model_name=args.backend_model,
            dataset_name=args.dataset_name,
            experiment_name=experiment_name,
            max_workers=args.max_workers,
            num_trials=args.num_trials,
            use_memory=True,
            use_memory_addition=args.use_memory_addition,
            use_memory_deletion=args.use_memory_deletion,
            rewrite_on_failure=args.rewrite_on_failure,
            delete_freq=args.delete_freq,
            freq_threshold=args.freq_threshold,
            utility_threshold=args.utility_threshold,
            workspace_id=args.memory_workspace_id,
            api_url=args.memory_api_url,
            batch_size=args.batch_size,
            temperature=resolved_temperature,
        )
    dump_final_memory_and_manifest(starting_memory_path=starting_memory_path)


if __name__ == "__main__":
    main()
