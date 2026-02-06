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
        choices=["wo_mem", "w_mem_cold", "w_mem_warm"],
        help=(
            "Benchmark mode: wo_mem (no memory), "
            "w_mem_cold (fresh empty workspace; accumulate online), "
            "w_mem_warm (load provided starting memory, then evaluate and keep accumulating)"
        ),
    )

    parser.add_argument(
        "--backend-model", default="gpt-oss-120b", help="LLM backend model name"
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
):
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
):
    path: Path = Path(f"./exp_result/{model_name}")
    path.mkdir(parents=True, exist_ok=True)

    task_ids = load_task_ids(dataset_name)
    result: list = []

    def dump_file():
        with open(path / f"{experiment_name}.jsonl", "a") as f:
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

    if args.mode not in {"wo_mem", "w_mem_cold", "w_mem_warm"}:
        raise ValueError(f"Unsupported mode: {args.mode}")

    experiment_name = args.experiment_name
    if not experiment_name:
        experiment_name = f"{args.dataset_name}_{args.mode}"

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
        )


if __name__ == "__main__":
    main()
