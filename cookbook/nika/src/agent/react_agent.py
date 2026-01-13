import asyncio
import logging
import os
import re
from typing import Any

import langsmith as ls
import requests
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError
from typing_extensions import TypedDict

from agent.domain_agents.diagnosis_agent import DiagnosisAgent
from agent.domain_agents.submission_agent import SubmissionAgent
from agent.utils.loggers import FileLoggerHandler
from nika.utils.logger import system_logger
from nika.utils.session import Session

load_dotenv()


logging.basicConfig(level=logging.INFO)


class AgentState(TypedDict):
    """The state of the agent."""

    messages: list[BaseMessage]
    diagnosis_report: str = Field(
        default="",
        description="The diagnosis report of the network state after analysis.",
    )
    is_max_steps_reached: bool = Field(
        default=False,
        description="Indicates whether the agent has reached the maximum number of steps allowed.",
    )


class BasicReActAgent:
    def __init__(
        self,
        backend_model,
        max_steps: int = 20,
        use_memory: bool = False,
        use_memory_addition: bool = False,
        use_memory_deletion: bool = False,
        freq_threshold: int = 5,
        utility_threshold: float = 0.5,
        memory_base_url: str = "http://0.0.0.0:8002/",
        memory_workspace_id: str = "nika_v1",
    ):
        self.max_steps = max_steps
        self.use_memory = use_memory
        self.use_memory_addition = use_memory_addition if use_memory else False
        self.use_memory_deletion = use_memory_deletion if use_memory else False
        self.freq_threshold = freq_threshold
        self.utility_threshold = utility_threshold
        self.memory_base_url = memory_base_url
        self.memory_workspace_id = memory_workspace_id
        self.retrieved_memory_list: list[Any] = []

        # Set up Langfuse callback handler
        # Initialize Langfuse client
        langfuse = get_client()

        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        self.langfuse_handler = CallbackHandler()

        if langfuse.auth_check():
            system_logger.info("Authentication to Langfuse successful.")
        else:
            system_logger.warning("Authentication to Langfuse failed. Please check your LANGFUSE_API_KEY.")

        # load agent and tools
        diagnosis_agent = DiagnosisAgent(backend_model=backend_model)
        asyncio.run(diagnosis_agent.load_tools())
        self.diagnosis_agent = diagnosis_agent.get_agent()

        submission_agent = SubmissionAgent(backend_model=backend_model)
        asyncio.run(submission_agent.load_tools())
        self.submission_agent = submission_agent.get_agent()

        # build the state graph
        worker_builder = StateGraph(AgentState)
        worker_builder.add_node("diagnosis_agent", self.diagnosis_agent_builder)
        worker_builder.add_node("submission_agent", self.submission_agent_builder)

        worker_builder.add_edge(START, "diagnosis_agent")
        worker_builder.add_conditional_edges(
            "diagnosis_agent",
            lambda state: state.get("is_max_steps_reached", False),
            {
                True: END,
                False: "submission_agent",
            },
        )

        worker_builder.add_edge("submission_agent", END)

        # compile the graph
        self.graph = worker_builder.compile()

    def load_session(self):
        self.session = Session()
        self.session.load_running_session()

    async def run(
        self,
        task_description: str,
        after_score: float = 0.0,
        previous_memories: list | None = None,
    ):
        """
        Run the agent with optional previous memories from failed attempts.

        Args:
            task_description: The task description to solve.
            after_score: Score from previous evaluation (for memory utility update).
            previous_memories: Memories from previous failed attempts (for retry logic).
        """
        self.load_session()

        # Retrieve memory before running the agent
        enriched_task_description = task_description
        if self.use_memory:
            if previous_memories and len(previous_memories) > 0:
                # Use previous memories from failed attempts (retry scenario)
                formatted_memories = []
                for i, memory in enumerate(previous_memories, 1):
                    condition = memory.get("when_to_use", "")
                    memory_content = memory.get("content", "")
                    memory_text = f"Experience {i}:\n  When to use: {condition}\n  Content: {memory_content}\n"
                    formatted_memories.append(memory_text)
                enriched_task_description = (
                    task_description
                    + "\n\nSome Related Experience to help you complete the task:\n"
                    + "\n".join(formatted_memories)
                )
                system_logger.info(f"Using {len(previous_memories)} previous memories from failed attempts")
            else:
                # First attempt: retrieve from ReMe
                memory_response = self.get_memory(task_description)
                if memory_response and "answer" in memory_response:
                    self.retrieved_memory_list = memory_response.get("metadata", {}).get("memory_list", [])
                    task_memory = memory_response["answer"]
                    system_logger.info(f"Retrieved task memory: {task_memory}")
                    enriched_task_description = (
                        task_description
                        + "\n\nSome Related Experience to help you complete the task:\n"
                        + re.sub(r"(?i)\bMemory\s*(\d+)\s*[:]", r"Experience \1:", task_memory)
                    )

        with ls.tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": self.session.scenario_name,
                "problem": self.session.problem_names[0],
                "topo_size": self.session.scenario_topo_size,
                "backend_model": self.session.backend_model,
            },
        ):
            result = await self.graph.ainvoke(
                {
                    "messages": [HumanMessage(content=enriched_task_description)],
                },
                config={"callbacks": [self.langfuse_handler]},
            )

        # Store trajectory as memory after task completion
        if self.use_memory:
            # Update memory usage information
            update_utility = after_score == 1.0
            if self.retrieved_memory_list:
                self.update_memory_information(self.retrieved_memory_list, update_utility)

        return result

    def store_memory_from_result(self, task_id: str, task_history: list, score: float):
        """Store trajectory as memory after successful task completion."""
        if not self.use_memory:
            return []

        trajectory = self.get_traj_from_messages(task_id, task_history, score)
        new_memories = self.add_memory([trajectory])

        # If task failed, delete the newly created memories
        if score != 1.0 and new_memories:
            self.delete_memory_by_ids([mem["memory_id"] for mem in new_memories])
            return []

        return new_memories

    async def diagnosis_agent_builder(self, state: AgentState):
        try:
            diagnosis_report = await self.diagnosis_agent.ainvoke(
                {"messages": state["messages"]},
                config={
                    "callbacks": [FileLoggerHandler(name="diagnosis_agent")],
                    "recursion_limit": self.max_steps,
                },
                debug=True,
            )
            return {"diagnosis_report": [diagnosis_report["messages"][-1].content], "is_max_steps_reached": False}
        except ValidationError as e:
            FileLoggerHandler(name="diagnosis_agent").logger.error(f"Diagnosis agent validation error: {e}")
            return {
                "messages": [HumanMessage(content=f"Error: {e}")],
                "diagnosis_report": ["ERROR_VALIDATION"],
                "is_max_steps_reached": False,
            }
        except GraphRecursionError:
            FileLoggerHandler(name="diagnosis_agent")._log(
                event_type="error",
                payload={"message": "Diagnosis agent reached max recursion limit."},
            )
            return {
                "messages": [HumanMessage(content="Error: diagnosis did not finish within max steps.")],
                "diagnosis_report": ["ERROR_MAX_STEPS_REACHED"],
                "is_max_steps_reached": True,
            }

    async def submission_agent_builder(self, state: AgentState):
        diag_text = state["diagnosis_report"][-1]
        result = await self.submission_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Based on the diagnosis report: {diag_text}, please provide the submission. Do not submit if no report available."
                    ),
                ]
            },
            config={
                "callbacks": [FileLoggerHandler(name="submission_agent")],
                "recursion_limit": self.max_steps,
            },
            debug=True,
        )
        return {
            "messages": result["messages"],
        }

    # ================== ReMe Memory API Methods ==================

    def handle_api_response(self, response: requests.Response) -> dict | None:
        """Handle API response with proper error checking."""
        if response.status_code != 200:
            system_logger.error(f"ReMe API Error: {response.status_code} - {response.text}")
            return None
        return response.json()

    def get_memory(self, query: str) -> dict | None:
        """Retrieve relevant task memories based on a query."""
        try:
            response = requests.post(
                url=f"{self.memory_base_url}retrieve_task_memory",
                json={
                    "workspace_id": self.memory_workspace_id,
                    "query": query,
                },
            )
            result = self.handle_api_response(response)
            if result:
                system_logger.info(f"Memory retrieved for query: {query[:100]}...")
            return result
        except requests.RequestException as e:
            system_logger.error(f"Failed to retrieve memory: {e}")
            return None

    def add_memory(self, trajectories: list) -> list:
        """Generate a summary of conversation messages and create task memories."""
        try:
            response = requests.post(
                url=f"{self.memory_base_url}summary_task_memory",
                json={
                    "workspace_id": self.memory_workspace_id,
                    "trajectories": trajectories,
                },
            )
            result = self.handle_api_response(response)
            if not result:
                return []

            memory_list = result.get("metadata", {}).get("memory_list", [])
            system_logger.info(f"Task memory created: {len(memory_list)} memories")
            return memory_list
        except requests.RequestException as e:
            system_logger.error(f"Failed to add memory: {e}")
            return []

    def delete_memory_by_ids(self, memory_ids: list):
        """Delete specific memories by their IDs."""
        try:
            response = requests.post(
                url=f"{self.memory_base_url}vector_store",
                json={
                    "workspace_id": self.memory_workspace_id,
                    "action": "delete_ids",
                    "memory_ids": memory_ids,
                },
            )
            response.raise_for_status()
            system_logger.info(f"Deleted {len(memory_ids)} memories")
        except requests.RequestException as e:
            system_logger.error(f"Failed to delete memories: {e}")

    def update_memory_information(self, memory_list: list, update_utility: bool = False):
        """Update memory usage records."""
        try:
            response = requests.post(
                url=f"{self.memory_base_url}record_task_memory",
                json={
                    "workspace_id": self.memory_workspace_id,
                    "memory_dicts": memory_list,
                    "update_utility": update_utility,
                },
            )
            response.raise_for_status()
            system_logger.info(f"Updated memory information: {response.json()}")
        except requests.RequestException as e:
            system_logger.error(f"Failed to update memory information: {e}")

    def delete_memory(self):
        """Delete low-quality memories based on frequency and utility thresholds."""
        try:
            response = requests.post(
                url=f"{self.memory_base_url}delete_task_memory",
                json={
                    "workspace_id": self.memory_workspace_id,
                    "freq_threshold": self.freq_threshold,
                    "utility_threshold": self.utility_threshold,
                },
            )
            response.raise_for_status()
            system_logger.info("Deleted low-quality memories based on thresholds")
        except requests.RequestException as e:
            system_logger.error(f"Failed to delete low-quality memories: {e}")

    def get_traj_from_messages(self, task_id: str, messages: list, score: float) -> dict:
        """Convert message history to trajectory format for memory storage."""
        # Remove the injected experience from the task description
        cleaned_messages = []
        for msg in messages:
            if hasattr(msg, "content"):
                content = msg.content
                # Remove injected experience section
                pattern = r"\n\nSome Related Experience to help you complete the task:.*"
                content = re.sub(pattern, "", content, flags=re.DOTALL)
                cleaned_messages.append({"role": msg.type, "content": content})
            elif isinstance(msg, dict):
                content = msg.get("content", "")
                pattern = r"\n\nSome Related Experience to help you complete the task:.*"
                content = re.sub(pattern, "", content, flags=re.DOTALL)
                cleaned_messages.append({"role": msg.get("role", "user"), "content": content})

        return {
            "task_id": task_id,
            "messages": cleaned_messages,
            "score": score,
        }
