# flake8: noqa: E402
import os
import time
import argparse

import ray
import requests
from ray import logger
from tqdm import tqdm

os.environ["APPWORLD_ROOT"] = "."
from dotenv import load_dotenv

load_dotenv("../../.env")

import json
from pathlib import Path

from appworld import load_task_ids

from appworld_react_agent import AppworldReactAgent


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


def dump_memory(workspace_id: str, path: str = "./", api_url: str = "http://0.0.0.0:8002/"):
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


def load_memory(workspace_id: str, path: str = "docs/library", api_url: str = "http://0.0.0.0:8002/"):
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
    experiment_suffix: str,
    max_workers: int,
    num_trials: int = 1,
    use_memory: bool = False,
    use_memory_addition: bool = False,
    use_memory_deletion: bool = False,
    delete_freq: int = 10,
    freq_threshold: int = 5,
    utility_threshold: float = 0.5,
    workspace_id: str = "appworld_v1",
    api_url: str = "http://0.0.0.0:8002/",
    batch_size: int = 4,
    # Azure OpenAI configuration
    llm_api_key: str | None = None,
    llm_api_base: str | None = None,
    llm_api_version: str | None = None,
):
    experiment_name = dataset_name + "_" + experiment_suffix
    path: Path = Path(f"./exp_result/{model_name}")
    path.mkdir(parents=True, exist_ok=True)

    task_ids = load_task_ids(dataset_name)
    result: list = []

    def dump_file():
        with open(path / f"{experiment_name}.jsonl", "a") as f:
            for x in result:
                f.write(json.dumps(x) + "\n")

    # Track statistics for progress bar
    success_count = 0
    total_trials_used = 0

    if max_workers > 1:
        # Process tasks in batches
        total_tasks = len(task_ids)
        num_batches = (total_tasks + batch_size - 1) // batch_size  # Ceiling division

        logger.info(f"Total tasks: {total_tasks}, Batch size: {batch_size}, Number of batches: {num_batches}")

        pbar = tqdm(total=total_tasks, desc="Tasks", unit="task")

        for batch_idx in range(num_batches):
            # Initialize Ray for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_tasks)
            batch_task_ids = task_ids[start_idx:end_idx]

            logger.info(f"Starting batch {batch_idx + 1}/{num_batches} with {len(batch_task_ids)} tasks")

            # Initialize Ray with the number of CPUs needed for this batch
            ray.init(num_cpus=len(batch_task_ids))

            future_list: list = []
            for i, task_id in enumerate(batch_task_ids):
                actor = AppworldReactAgent.remote(
                    index=start_idx+i,
                    model_name=model_name,
                    task_ids=[task_id],
                    experiment_name=experiment_name,
                    num_trials=num_trials,
                    use_memory=use_memory,
                    use_memory_addition=use_memory_addition,
                    use_memory_deletion=use_memory_deletion,
                    delete_freq=delete_freq,
                    freq_threshold=freq_threshold,
                    utility_threshold=utility_threshold,
                    memory_workspace_id=workspace_id,
                    memory_base_url=api_url,
                    # Azure OpenAI configuration
                    api_key=llm_api_key,
                    api_base=llm_api_base,
                    api_version=llm_api_version,
                )
                future = actor.execute.remote()
                future_list.append(future)
                time.sleep(1)

            logger.info(f"Batch {batch_idx + 1} submit complete, waiting for results...")

            # Collect results from this batch
            for i, (task_id, future) in enumerate(zip(batch_task_ids, future_list)):
                try:
                    t_result = ray.get(future)
                    if t_result:
                        if isinstance(t_result, list):
                            result.extend(t_result)
                            # Update statistics from the last trial result
                            last_result = t_result[-1] if t_result else None
                            if last_result:
                                trials_used = last_result.get("run_id", 0) + 1
                                total_trials_used += trials_used
                                if last_result.get("after_score", 0) == 1:
                                    success_count += 1
                        else:
                            result.append(t_result)
                            trials_used = t_result.get("run_id", 0) + 1
                            total_trials_used += trials_used
                            if t_result.get("after_score", 0) == 1:
                                success_count += 1
                except Exception as e:
                    logger.exception(f"run ray error with task_id={task_id}")

                pbar.update(1)
                pbar.set_postfix({
                    "success": f"{success_count}/{pbar.n}",
                    "avg_trials": f"{total_trials_used/pbar.n:.1f}/{num_trials}"
                })

            # Shutdown Ray to free resources before next batch
            ray.shutdown()
            logger.info(f"Batch {batch_idx + 1}/{num_batches} complete, Ray resources released")

            # Optional: small delay between batches
            if batch_idx < num_batches - 1:
                time.sleep(2)

        pbar.close()
        dump_file()

    else:
        # Single worker mode - still use Ray for consistency
        pbar = tqdm(task_ids, desc="Tasks", unit="task")
        for index, task_id in enumerate(pbar):
            actor = AppworldReactAgent.remote(
                index=index,
                model_name=model_name,
                task_ids=[task_id],
                experiment_name=experiment_name,
                num_trials=num_trials,
                use_memory=use_memory,
                use_memory_addition=use_memory_addition,
                use_memory_deletion=use_memory_deletion,
                delete_freq=delete_freq,
                freq_threshold=freq_threshold,
                utility_threshold=utility_threshold,
                memory_workspace_id=workspace_id,
                memory_base_url=api_url,
                # Azure OpenAI configuration
                api_key=llm_api_key,
                api_base=llm_api_base,
                api_version=llm_api_version,
            )
            task_results = ray.get(actor.execute.remote())
            if isinstance(task_results, list):
                result.extend(task_results)
                # Update statistics from the last trial result
                last_result = task_results[-1] if task_results else None
                if last_result:
                    trials_used = last_result.get("run_id", 0) + 1
                    total_trials_used += trials_used
                    if last_result.get("after_score", 0) == 1:
                        success_count += 1
            else:
                result.append(task_results)
                trials_used = task_results.get("run_id", 0) + 1
                total_trials_used += trials_used
                if task_results.get("after_score", 0) == 1:
                    success_count += 1

            pbar.set_postfix({
                "success": f"{success_count}/{index+1}",
                "avg_trials": f"{total_trials_used/(index+1):.1f}/{num_trials}"
            })

        pbar.close()
        dump_file()

def main():
    parser = argparse.ArgumentParser(description="Run AppWorld experiments with ReMe memory")
    parser.add_argument("--use-memory", action="store_true", help="Enable ReMe memory system")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of concurrent tasks per batch")
    parser.add_argument("--num-trials", type=int, default=2, help="Number of trials per task")
    parser.add_argument("--dataset-name", type=str, default="test_normal", help="Dataset name")
    args = parser.parse_args()

    max_workers = args.max_workers
    num_runs = args.num_runs
    batch_size = args.batch_size
    num_trials = args.num_trials

    # Azure OpenAI model configuration (read from environment variables)
    model_name = os.getenv("AZURE_LLM_DEPLOYMENT_NAME", "gpt-5.2")
    llm_api_base = os.getenv("AZURE_LLM_API_BASE")
    llm_api_version = os.getenv("AZURE_LLM_API_VERSION", "2024-02-01")
    # API key is read from environment variable: AZURE_LLM_API_KEY or AZURE_API_KEY

    use_memory = args.use_memory
    use_memory_addition = args.use_memory
    use_memory_deletion = args.use_memory
    workspace_id = "appworld"
    api_url = "http://0.0.0.0:8002/"

    experiment_suffix = "with-memory" if use_memory else "without-memory"

    if use_memory:
        # Clean up workspace before starting
        logger.info("Deleting workspace...")
        delete_workspace(workspace_id=workspace_id, api_url=api_url)
        time.sleep(5)

        # First run to build task memories
        logger.info("Start load experiments to build task memories")
        load_memory(workspace_id=workspace_id, api_url=api_url)

    for i in range(num_runs):
        run_agent(
            model_name=model_name,
            dataset_name=args.dataset_name,
            experiment_suffix=experiment_suffix,
            max_workers=max_workers,
            num_trials=num_trials,
            use_memory=use_memory,
            use_memory_addition=use_memory_addition,
            use_memory_deletion=use_memory_deletion,
            delete_freq=5,
            freq_threshold=5,
            utility_threshold=0.5,
            workspace_id=workspace_id,
            api_url=api_url,
            batch_size=batch_size,
            # Azure OpenAI configuration
            llm_api_base=llm_api_base,
            llm_api_version=llm_api_version,
        )

if __name__ == "__main__":
    main()
