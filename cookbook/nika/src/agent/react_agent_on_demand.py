import asyncio
import json
import logging
import os
import re
from typing import Any

import langsmith as ls
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError
from typing_extensions import TypedDict

from agent.domain_agents.diagnosis_agent_on_demand import DiagnosisAgentOnDemand
from agent.domain_agents.submission_agent import SubmissionAgent
from agent.react_agent import BasicReActAgent
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


class OnDemandReActAgent(BasicReActAgent):
    """ReAct agent that retrieves experiences only when the LLM calls a tool."""

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
        temperature: float | None = None,
        use_memory_retrieval: bool | None = None,
        enable_memory_store: bool | None = None,
        enable_memory_tool: bool = True,
    ):
        self.max_steps = max_steps
        self.use_memory = use_memory
        self.use_memory_retrieval = use_memory if use_memory_retrieval is None else use_memory_retrieval
        self.use_memory_addition = use_memory_addition if use_memory else False
        self.use_memory_deletion = use_memory_deletion if use_memory else False
        self.enable_memory_store = (
            (use_memory_addition or use_memory) if enable_memory_store is None else enable_memory_store
        )
        self.freq_threshold = freq_threshold
        self.utility_threshold = utility_threshold
        self.memory_base_url = memory_base_url
        self.memory_workspace_id = memory_workspace_id
        self.retrieved_memory_list: list[Any] = []
        self.memory_tool_responses: list[dict] = []
        self.task_history: list[Any] = []
        self.temperature = temperature
        self.enable_memory_tool = enable_memory_tool

        # Set up Langfuse callback handler
        langfuse = get_client()
        self.langfuse_handler = CallbackHandler()

        if langfuse.auth_check():
            system_logger.info("Authentication to Langfuse successful.")
        else:
            system_logger.warning("Authentication to Langfuse failed. Please check your LANGFUSE_API_KEY.")

        # load agent and tools
        diagnosis_agent = DiagnosisAgentOnDemand(
            backend_model=backend_model,
            temperature=temperature,
            memory_base_url=memory_base_url,
            memory_workspace_id=memory_workspace_id,
            enable_memory_tool=enable_memory_tool,
            memory_retrieval_callback=self._record_memory_response,
        )
        asyncio.run(diagnosis_agent.load_tools())
        self.diagnosis_agent = diagnosis_agent.get_agent()

        submission_agent = SubmissionAgent(backend_model=backend_model, temperature=temperature)
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

    def _record_memory_response(self, response: dict) -> None:
        if not isinstance(response, dict):
            return
        self.memory_tool_responses.append(response)
        memory_list = response.get("metadata", {}).get("memory_list", [])
        if memory_list:
            self.retrieved_memory_list = memory_list

    def load_session(self):
        self.session = Session()
        self.session.load_running_session()

    def reset_task_history(self):
        self.task_history = []

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        try:
            return json.dumps(content, ensure_ascii=True, default=str)
        except TypeError:
            return str(content)

    def _normalize_role(self, role: Any) -> str:
        if role is None:
            return "user"
        role_value = role.value if hasattr(role, "value") else str(role)
        role_value = role_value.lower()
        if role_value == "human":
            return "user"
        if role_value == "ai":
            return "assistant"
        if role_value in {"system", "user", "assistant", "tool"}:
            return role_value
        return "user"

    def _message_key(self, message: Any) -> tuple[str | None, str]:
        if hasattr(message, "content"):
            role = self._normalize_role(getattr(message, "type", None))
            content = self._normalize_content(message.content)
            return (role, content)
        if isinstance(message, dict):
            role = self._normalize_role(message.get("role"))
            return (role, self._normalize_content(message.get("content", "")))
        return (None, str(message))

    def _extend_task_history(self, messages: list[Any]):
        if not messages:
            return
        for message in messages:
            if self.task_history and self._message_key(self.task_history[-1]) == self._message_key(message):
                continue
            self.task_history.append(message)

    def _update_memory_debug_info(self):
        if not hasattr(self.session, "session_dir"):
            return
        memory_debug_path = os.path.join(self.session.session_dir, "memory_debug.json")
        try:
            if os.path.exists(memory_debug_path):
                with open(memory_debug_path, "r") as f:
                    info = json.load(f)
            else:
                info = {}
        except Exception:
            info = {}

        info["memory_tool_calls"] = len(self.memory_tool_responses)
        if self.memory_tool_responses:
            last_response = self.memory_tool_responses[-1]
            info["memory_tool_last_query"] = last_response.get("query")
            info["memory_tool_last_reason"] = last_response.get("reason")
            memory_list = last_response.get("metadata", {}).get("memory_list", [])
            info["memory_tool_last_memory_ids"] = [
                mem.get("memory_id") for mem in memory_list if isinstance(mem, dict) and mem.get("memory_id")
            ]
            embedding_scores = []
            for mem in memory_list:
                if not isinstance(mem, dict):
                    continue
                score = mem.get("embedding_score")
                if score is None:
                    score = mem.get("metadata", {}).get("embedding_score")
                if score is None:
                    score = mem.get("metadata", {}).get("_score")
                if score is None:
                    continue
                embedding_scores.append(
                    {
                        "memory_id": mem.get("memory_id"),
                        "embedding_score": score,
                    }
                )
            if embedding_scores:
                info["memory_tool_last_embedding_scores"] = embedding_scores

        with open(memory_debug_path, "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False, default=str)

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
        self.reset_task_history()

        # On-demand mode: do not auto-retrieve memory at the start.
        enriched_task_description = task_description
        memory_debug_info = {
            "use_memory": self.use_memory,
            "memory_mode": "on_demand_tool",
            "enable_memory_tool": self.enable_memory_tool,
            "original_task_description": task_description,
            "memory_response": None,
            "retrieved_memory_list": [],
            "previous_memories": None,
            "enriched_task_description": None,
        }

        if self.use_memory and self.use_memory_retrieval:
            if previous_memories and len(previous_memories) > 0:
                memory_debug_info["previous_memories"] = previous_memories
                formatted_memories = []
                for i, memory in enumerate(previous_memories, 1):
                    memory_content = memory.get("content", "")
                    memory_text = f"{memory_content}\n"
                    formatted_memories.append(memory_text)
                enriched_task_description = (
                    task_description
                    + "\n\nSome Related Experience to help you complete the task:\n"
                    + "\n".join(formatted_memories)
                )
                system_logger.info(f"Using {len(previous_memories)} previous memories from failed attempts")

        memory_debug_info["enriched_task_description"] = enriched_task_description

        # Save memory debug info to session directory
        if hasattr(self.session, "session_dir"):
            os.makedirs(self.session.session_dir, exist_ok=True)
            memory_debug_path = os.path.join(self.session.session_dir, "memory_debug.json")
            with open(memory_debug_path, "w") as f:
                json.dump(memory_debug_info, f, indent=2, ensure_ascii=False, default=str)
            system_logger.info(f"Memory debug info saved to {memory_debug_path}")

        input_messages = [HumanMessage(content=enriched_task_description)]
        self._extend_task_history(input_messages)
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
                    "messages": input_messages,
                },
                config={"callbacks": [self.langfuse_handler]},
            )

        # Store trajectory as memory after task completion
        if self.use_memory:
            update_utility = after_score == 1.0
            if self.retrieved_memory_list:
                self.update_memory_information(self.retrieved_memory_list, update_utility)

        self._update_memory_debug_info()
        return result

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
            self._extend_task_history(diagnosis_report.get("messages", []))
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
                        content=(
                            "Based on the diagnosis report: "
                            f"{diag_text}, please provide the submission. "
                            "Do not submit if no report available."
                        )
                    ),
                ]
            },
            config={
                "callbacks": [FileLoggerHandler(name="submission_agent")],
                "recursion_limit": self.max_steps,
            },
            debug=True,
        )
        self._extend_task_history(result.get("messages", []))
        return {
            "messages": result["messages"],
        }
