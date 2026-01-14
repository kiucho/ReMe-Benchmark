import argparse
import json
import random

from nika.orchestrator.problems.prob_pool import get_problem_instance, list_avail_problem_names
from nika.orchestrator.problems.problem_base import TaskLevel
from nika.utils.logger import system_logger
from nika.utils.session import Session


def inject_failure(problem_names: list[str], re_inject: bool = True, seed: int | None = None):
    """
    Inject failure into the network environment based on the root cause name.
    
    Args:
        problem_names: List of problem names to inject.
        re_inject: Whether to re-inject the fault.
        seed: Optional seed for random fault injection. If None, uses session_id[-4:].
              Use the same seed across experiments for reproducible comparisons.
    """
    logger = system_logger

    session = Session()
    session.load_running_session()
    # save session data for follow-up steps
    session.update_session("problem_names", problem_names)

    for problem_name in problem_names:
        # check if problem_name in the available problems
        if problem_name not in list_avail_problem_names():
            raise ValueError(f"Unknown problem name: {problem_name}")

    scenario_params = session.scenario_params if hasattr(session, "scenario_params") else {}

    # Determine the seed to use
    if seed is not None:
        actual_seed = seed
        logger.info(f"Using user-specified seed: {seed}")
    else:
        actual_seed = session.session_id[-4:]
        logger.info(f"Using session-based seed: {actual_seed}")
    
    # Save seed to session for reproducibility tracking
    session.update_session("random_seed", actual_seed)

    tot_tasks = []
    for task_level in TaskLevel:
        random.seed(actual_seed)
        problem = get_problem_instance(
            problem_names=problem_names, task_level=task_level, scenario_name=session.scenario_name, **scenario_params
        )
        tot_tasks.append(problem)

    if re_inject:
        tot_tasks[0].inject_fault()

    logger.info(f"Session {session.session_id}, injected problem(s): {problem_names} under {session.scenario_name}.")
    task_description = problem.get_task_description()

    session.update_session("task_description", task_description)

    # save the ground truth for evaluation
    tot_gt = {}
    for problem in tot_tasks:
        gt = problem.get_submission().model_dump_json()
        tot_gt.update(json.loads(gt))

    session.write_gt(tot_gt)
    logger.info(f"Ground truth saved for session ID: {session.session_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject failure into the network environment.")

    parser.add_argument(
        "--problem",
        type=str,
        default="frr_service_down",
        help="The issue to inject, e.g. frr_service_down, bmv2_service_down, etc.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible fault injection. Use the same seed across experiments for fair comparison.",
    )
    args = parser.parse_args()

    inject_failure(problem_names=[args.problem], seed=args.seed)
