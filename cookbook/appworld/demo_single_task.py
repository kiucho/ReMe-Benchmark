#!/usr/bin/env .venv-appworld/bin/python
# flake8: noqa: E402, E501
"""
ReMe Self-Evolving Memory Demo Script

This script demonstrates the impact of ReMe memory on AppWorld task performance by:
1. Running a task WITHOUT memory
2. Running the same task WITH memory (all memory features enabled by default)
3. Showing detailed comparison of:
   - Retrieved memories (raw from vector store)
   - Reranked memories (after LLM reranking)
   - Rewritten context (final experience text)
   - Full execution trajectory
   - Added memories (self-evolving memory)
   - Failure-aware reflection (if num_trials > 1)

When using memory (--compare or --with-memory-only), all memory features are ENABLED by default:
- Memory retrieval & reranking & rewriting
- Memory addition (new experiences added to pool after successful tasks)
- Memory update (freq/utility attributes updated for retrieved memories)
- Memory deletion (low-utility memories removed)

Failure-Aware Reflection (--num-trials > 1):
- If first attempt fails, extract memories from failed trajectory
- Use those memories in next attempt
- Repeat until success or max trials reached

Usage:
    # Compare mode: runs both without/with memory (with-memory uses all features)
    ./demo_single_task.py --task-id 3d9a636_1 --compare
    
    # Run only with memory (all features enabled)
    ./demo_single_task.py --task-id 3d9a636_1 --with-memory-only
    
    # Enable failure-aware reflection with 3 retries
    ./demo_single_task.py --task-id 3d9a636_1 --with-memory-only --num-trials 3
    
    # Run only without memory (baseline)
    ./demo_single_task.py --task-id 3d9a636_1 --without-memory-only
    
    # Disable specific memory features if needed
    ./demo_single_task.py --task-id 3d9a636_1 --with-memory-only --disable-memory-addition
    ./demo_single_task.py --task-id 3d9a636_1 --with-memory-only --disable-memory-deletion
"""

import os
import sys
import argparse
import json
import datetime
import requests
from pathlib import Path

os.environ["APPWORLD_ROOT"] = "."
from dotenv import load_dotenv

load_dotenv("../../.env")

from loguru import logger

from appworld_react_agent_verbose import (
    AppworldReactAgentVerbose,
    ExecutionResult,
    print_execution_result,
    save_result_to_json,
)


def check_reme_server(api_url: str = "http://0.0.0.0:8002/") -> bool:
    """Check if ReMe server is running."""
    try:
        response = requests.get(f"{api_url}health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def load_memory_library(workspace_id: str, api_url: str = "http://0.0.0.0:8002/"):
    """Load pre-built memory library into vector store."""
    print("\n[INFO] Loading memory library...")
    
    try:
        response = requests.post(
            url=f"{api_url}vector_store",
            json={
                "workspace_id": workspace_id,
                "action": "load",
                "path": "docs/library",
            },
        )
        
        if response.status_code == 200:
            print("[INFO] Memory library loaded successfully")
            return True
        else:
            print(f"[WARN] Failed to load memory library: {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Error loading memory library: {e}")
        return False


def run_task_without_memory(
    task_id: str,
    experiment_name: str,
) -> ExecutionResult:
    """Run a task without ReMe memory."""
    print("\n" + "=" * 80)
    print("Running task WITHOUT ReMe Memory...")
    print("=" * 80)
    
    agent = AppworldReactAgentVerbose(
        task_id=task_id,
        experiment_name=f"{experiment_name}_without_memory",
        use_memory=False,
    )
    
    return agent.execute()


def run_task_with_memory(
    task_id: str,
    experiment_name: str,
    workspace_id: str = "appworld",
    api_url: str = "http://0.0.0.0:8002/",
    num_trials: int = 1,
    use_memory_addition: bool = False,
    use_memory_deletion: bool = False,
    freq_threshold: int = 5,
    utility_threshold: float = 0.5,
) -> ExecutionResult:
    """Run a task with ReMe memory."""
    print("\n" + "=" * 80)
    print("Running task WITH ReMe Memory...")
    if num_trials > 1:
        print(f"  → Failure-Aware Reflection: ENABLED ({num_trials} max trials)")
    if use_memory_addition:
        print("  → Memory Addition: ENABLED (new memories will be added to experience pool)")
    if use_memory_deletion:
        print("  → Memory Deletion: ENABLED (low-utility memories will be removed)")
    print("=" * 80)
    
    agent = AppworldReactAgentVerbose(
        task_id=task_id,
        experiment_name=f"{experiment_name}_with_memory",
        num_trials=num_trials,
        use_memory=True,
        use_memory_addition=use_memory_addition,
        use_memory_deletion=use_memory_deletion,
        freq_threshold=freq_threshold,
        utility_threshold=utility_threshold,
        memory_workspace_id=workspace_id,
        memory_base_url=api_url,
    )
    
    return agent.execute()


def print_comparison_summary(
    without_result: ExecutionResult,
    with_result: ExecutionResult,
):
    """Print a summary comparison of the two runs."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print("\n┌────────────────────────────────────────────────────────────────────────────┐")
    print("│                           SIDE-BY-SIDE COMPARISON                          │")
    print("├────────────────────────────────────────────────────────────────────────────┤")
    print(f"│ {'Metric':<25} │ {'Without Memory':<20} │ {'With Memory':<20} │")
    print("├────────────────────────────────────────────────────────────────────────────┤")
    
    # Score
    without_status = "SUCCESS" if without_result.score == 1.0 else f"FAILED ({without_result.score:.2f})"
    with_status = "SUCCESS" if with_result.score == 1.0 else f"FAILED ({with_result.score:.2f})"
    print(f"│ {'Result':<25} │ {without_status:<20} │ {with_status:<20} │")
    
    # Steps
    print(f"│ {'Total Steps':<25} │ {without_result.steps:<20} │ {with_result.steps:<20} │")
    
    # Task Completed
    without_completed = "Yes" if without_result.task_completed else "No"
    with_completed = "Yes" if with_result.task_completed else "No"
    print(f"│ {'Task Completed':<25} │ {without_completed:<20} │ {with_completed:<20} │")
    
    # Memory Stats
    if with_result.memory_pipeline:
        mp = with_result.memory_pipeline
        retrieved = len(mp.retrieved_memories)
        reranked = len(mp.reranked_memories)
        print(f"│ {'Memories Retrieved':<25} │ {'-':<20} │ {retrieved:<20} │")
        print(f"│ {'Memories After Rerank':<25} │ {'-':<20} │ {reranked:<20} │")
    
    # Memory Addition Stats
    if with_result.added_memories:
        added = len(with_result.added_memories)
        print(f"│ {'Memories Added':<25} │ {'-':<20} │ {added:<20} │")
    
    # Memory Update Stats
    updated_str = "Yes" if with_result.memory_updated else "No"
    print(f"│ {'Memory Info Updated':<25} │ {'-':<20} │ {updated_str:<20} │")
    
    # Failure-Aware Reflection Stats
    if with_result.num_trials > 1 or len(with_result.trial_results) > 1:
        trials_str = f"{len(with_result.trial_results)}/{with_result.num_trials}"
        print(f"│ {'Trials Attempted':<25} │ {'-':<20} │ {trials_str:<20} │")
    
    print("└────────────────────────────────────────────────────────────────────────────┘")
    
    # Failure-Aware Reflection Details
    if len(with_result.trial_results) > 1:
        print("\n📊 Failure-Aware Reflection Trial Details:")
        for trial in with_result.trial_results:
            status = "✅ SUCCESS" if trial.score == 1.0 else "❌ FAILED"
            reflection = " (used previous failure memories)" if trial.used_previous_memories else ""
            print(f"   Trial {trial.run_id + 1}: {status} (score: {trial.score:.2f}, steps: {trial.steps}){reflection}")
    
    # Improvement
    if with_result.score > without_result.score:
        improvement = (with_result.score - without_result.score) * 100
        print(f"\n✅ ReMe Memory improved task score by {improvement:.1f}%")
    elif with_result.score == without_result.score == 1.0:
        step_diff = without_result.steps - with_result.steps
        if step_diff > 0:
            print(f"\n✅ Both succeeded, but ReMe Memory saved {step_diff} steps")
        else:
            print("\n📊 Both runs achieved the same result")
    else:
        print("\n📊 Memory did not improve the result for this task")


def main():
    parser = argparse.ArgumentParser(
        description="ReMe Self-Evolving Memory Demo for AppWorld",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare with and without memory (recommended)
  python demo_single_task.py --task-id 3d9a636_1 --compare
  
  # Run only with memory (for debugging memory retrieval)
  python demo_single_task.py --task-id 3d9a636_1 --with-memory-only
  
  # Run only without memory (baseline)
  python demo_single_task.py --task-id 3d9a636_1 --without-memory-only
  
  # Save results to JSON
  python demo_single_task.py --task-id 3d9a636_1 --compare --output results.json
        """
    )
    
    parser.add_argument(
        "--task-id",
        type=str,
        default="3d9a636_1",
        help="AppWorld task ID to run (default: 3d9a636_1)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both with and without memory for comparison"
    )
    parser.add_argument(
        "--with-memory-only",
        action="store_true",
        help="Run only with memory enabled"
    )
    parser.add_argument(
        "--without-memory-only",
        action="store_true",
        help="Run only without memory (baseline)"
    )
    parser.add_argument(
        "--workspace-id",
        type=str,
        default="appworld",
        help="ReMe workspace ID for memory storage"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://0.0.0.0:8002/",
        help="ReMe API URL"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file base name (timestamp will be appended automatically, e.g., 'result' -> 'result_20260113_123456.json')"
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't append timestamp to output filename"
    )
    parser.add_argument(
        "--skip-memory-load",
        action="store_true",
        help="Skip loading memory library (if already loaded)"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=3,
        help="Number of trials for failure-aware reflection (default: 1, set > 1 to enable retry with learned memories)"
    )
    parser.add_argument(
        "--disable-memory-addition",
        action="store_true",
        help="Disable adding new memories to experience pool after task execution (enabled by default when using memory)"
    )
    parser.add_argument(
        "--disable-memory-deletion",
        action="store_true",
        help="Disable deletion of low-utility memories after task execution (enabled by default when using memory)"
    )
    parser.add_argument(
        "--freq-threshold",
        type=int,
        default=5,
        help="Frequency threshold for memory deletion (default: 5)"
    )
    parser.add_argument(
        "--utility-threshold",
        type=float,
        default=0.5,
        help="Utility threshold for memory deletion (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.compare, args.with_memory_only, args.without_memory_only]):
        print("[INFO] No mode specified, defaulting to --compare")
        args.compare = True
    
    # Generate experiment name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"demo_{args.task_id}_{timestamp}"
    
    # Print header
    print("\n" + "=" * 80)
    print("ReMe Self-Evolving Memory Demo")
    print("=" * 80)
    print(f"Task ID: {args.task_id}")
    print(f"Experiment: {experiment_name}")
    print(f"Mode: {'Compare' if args.compare else 'With Memory Only' if args.with_memory_only else 'Without Memory Only'}")
    
    # Show memory feature status for memory runs
    if args.compare or args.with_memory_only:
        print("\nMemory Features (for with-memory run):")
        print(f"  → Failure-Aware Reflection: {'ENABLED (' + str(args.num_trials) + ' trials)' if args.num_trials > 1 else 'DISABLED (single trial)'}")
        print(f"  → Memory Addition: {'ENABLED' if not args.disable_memory_addition else 'DISABLED'}")
        print(f"  → Memory Deletion: {'ENABLED' if not args.disable_memory_deletion else 'DISABLED'}")
        if not args.disable_memory_deletion:
            print(f"     - freq_threshold: {args.freq_threshold}")
            print(f"     - utility_threshold: {args.utility_threshold}")
    
    without_result = None
    with_result = None
    
    # Check if we need memory
    need_memory = args.compare or args.with_memory_only
    
    if need_memory:
        # Check ReMe server
        if not check_reme_server(args.api_url):
            print(f"\n[ERROR] ReMe server is not running at {args.api_url}")
            print("[INFO] Please start the ReMe server first:")
            print("       cd ReMe && python -m reme_ai.main")
            sys.exit(1)
        
        # Load memory library
        if not args.skip_memory_load:
            load_memory_library(args.workspace_id, args.api_url)
    
    # Run without memory
    if args.compare or args.without_memory_only:
        without_result = run_task_without_memory(args.task_id, experiment_name)
        print_execution_result(without_result, with_memory=False)
    
    # Run with memory (all memory features enabled by default)
    if args.compare or args.with_memory_only:
        with_result = run_task_with_memory(
            args.task_id,
            experiment_name,
            args.workspace_id,
            args.api_url,
            num_trials=args.num_trials,
            use_memory_addition=not args.disable_memory_addition,  # Enabled by default
            use_memory_deletion=not args.disable_memory_deletion,  # Enabled by default
            freq_threshold=args.freq_threshold,
            utility_threshold=args.utility_threshold,
        )
        print_execution_result(with_result, with_memory=True)
    
    # Print comparison if both were run
    if args.compare and without_result and with_result:
        print_comparison_summary(without_result, with_result)
    
    # Save to JSON
    output_dir = Path("./demo_results")
    output_dir.mkdir(exist_ok=True)
    
    if args.output:
        # User specified output filename
        output_base = Path(args.output)
        if args.no_timestamp:
            # Use as-is
            output_path = output_base
        else:
            # Append timestamp to filename (before extension)
            stem = output_base.stem  # filename without extension
            suffix = output_base.suffix or ".json"
            output_path = output_dir / f"{stem}_{timestamp}{suffix}"
    else:
        # Auto-generate output path with timestamp
        output_path = output_dir / f"{experiment_name}.json"
    
    save_result_to_json(without_result, with_result, str(output_path))
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
