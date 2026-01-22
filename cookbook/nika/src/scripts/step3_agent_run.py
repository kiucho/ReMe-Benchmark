import argparse
import asyncio
import logging

from agent.react_agent import BasicReActAgent
from agent.react_agent_on_demand import OnDemandReActAgent
from nika.utils.logger import system_logger
from nika.utils.session import Session


def _agent_selector(
    agent_type: str,
    backend_model: str,
    max_steps: int = 20,
    use_memory: bool = False,
    use_memory_addition: bool = False,
    use_memory_deletion: bool = False,
    freq_threshold: int = 5,
    utility_threshold: float = 0.5,
    memory_base_url: str = "http://0.0.0.0:8002/",
    memory_workspace_id: str = "nika_v1",
    temperature: float | None = None,
    use_memory_retrieval: bool | None = None,
    enable_memory_store: bool | None = None,
):
    match agent_type.lower():
        case "react":
            return BasicReActAgent(
                backend_model=backend_model,
                max_steps=max_steps,
                use_memory=use_memory,
                use_memory_addition=use_memory_addition,
                use_memory_deletion=use_memory_deletion,
                freq_threshold=freq_threshold,
                utility_threshold=utility_threshold,
                memory_base_url=memory_base_url,
                memory_workspace_id=memory_workspace_id,
                temperature=temperature,
                use_memory_retrieval=use_memory_retrieval,
                enable_memory_store=enable_memory_store,
            )
        case "react_on_demand":
            return OnDemandReActAgent(
                backend_model=backend_model,
                max_steps=max_steps,
                use_memory=use_memory,
                use_memory_addition=use_memory_addition,
                use_memory_deletion=use_memory_deletion,
                freq_threshold=freq_threshold,
                utility_threshold=utility_threshold,
                memory_base_url=memory_base_url,
                memory_workspace_id=memory_workspace_id,
                temperature=temperature,
                use_memory_retrieval=use_memory_retrieval,
                enable_memory_store=enable_memory_store,
            )
        case _:
            pass


def start_agent(
    agent_type: str,
    backend_model: str,
    max_steps: int,
    use_memory: bool = False,
    use_memory_addition: bool = False,
    use_memory_deletion: bool = False,
    freq_threshold: int = 5,
    utility_threshold: float = 0.5,
    memory_base_url: str = "http://0.0.0.0:8002/",
    memory_workspace_id: str = "nika_v1",
    previous_memories: list | None = None,
    temperature: float | None = None,
    use_memory_retrieval: bool | None = None,
    enable_memory_store: bool | None = None,
) -> BasicReActAgent | None:
    """Start the agent and return the agent instance for post-evaluation memory operations.

    Args:
        previous_memories: Memories from previous failed attempts (for num_trials retry logic).
    """
    logger = system_logger

    session = Session()
    session.load_running_session()
    session.update_session("agent_type", agent_type)
    session.update_session("backend_model", backend_model)
    session.start_session()

    logger.info(f"Starting agent: {agent_type} with backend {backend_model} in session {session.session_id}")
    agent = _agent_selector(
        agent_type,
        backend_model,
        max_steps=max_steps,
        use_memory=use_memory,
        use_memory_addition=use_memory_addition,
        use_memory_deletion=use_memory_deletion,
        freq_threshold=freq_threshold,
        utility_threshold=utility_threshold,
        memory_base_url=memory_base_url,
        memory_workspace_id=memory_workspace_id,
        temperature=temperature,
        use_memory_retrieval=use_memory_retrieval,
        enable_memory_store=enable_memory_store,
    )
    asyncio.run(agent.run(
        task_description=session.task_description,
        previous_memories=previous_memories,
    ))

    # stop session
    session.end_session()
    logger.info(f"Agent run completed for session ID: {session.session_id}")

    return agent  # Return agent for post-evaluation memory operations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=(
            "Run the specified agent to start troubleshooting. \n"
            "Note: the backend LLM must be configured before running this command."
        )
    )

    parser.add_argument(
        "--agent_type",
        type=str,
        default="ReAct",
        help="Type of agent to run (default: ReAct)",
    )

    parser.add_argument(
        "--backend_model",
        type=str,
        default="gpt-oss:20b",
        help="Backend model for the agent (default: gpt-oss:20b)",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="Maximum steps for the agent to take (default: 20)",
    )

    args = parser.parse_args()
    start_agent(args.agent_type, args.backend_model, args.max_steps)
