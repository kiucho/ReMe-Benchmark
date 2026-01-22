from textwrap import dedent
from typing import Callable, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools.structured import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field
import requests

from agent.llm.model_factory import load_model
from agent.utils.mcp_servers import MCPServerConfig
from nika.utils.logger import system_logger

load_dotenv()

ON_DEMAND_DIAGNOSIS_PROMPT = dedent("""\
    You are a network troubleshooting expert.
    Focus on (1) detecting if there is an anomaly, (2) localizing the faulty devices,
    and (3) identifying the root cause.

    Basic requirements:
    - Use the provided tools to gather necessary information.
    - Do not provide mitigation unless explicitly required.

    On-demand experience usage:
    - You have a tool `retrieve_experience` that searches past experiences.
    - Call it only when evidence is insufficient, conflicting, or you are stuck after
      using available tools at least once.
    - Do NOT call it at the start.
    - Before calling, summarize the current observations in 1-2 lines and use that as the query.
    - If experience content conflicts with observed evidence, trust observations first.
""").strip()


class RetrieveExperienceInput(BaseModel):
    query: str = Field(..., description="Concise summary of current symptoms and observations.")
    reason: str = Field(..., description="Why experience retrieval is needed now.")
    top_k: int = Field(3, ge=1, le=10, description="Number of experiences to retrieve.")


class DiagnosisAgentOnDemand:
    """Diagnosis agent that can retrieve experiences on-demand via a tool."""

    def __init__(
        self,
        backend_model: str = "gpt-oss:20b",
        temperature: float | None = None,
        memory_base_url: str = "http://0.0.0.0:8002/",
        memory_workspace_id: str = "nika_v1",
        enable_memory_tool: bool = True,
        memory_retrieval_callback: Optional[Callable[[dict], None]] = None,
    ):
        mcp_server_config = MCPServerConfig().load_config(if_submit=False)
        self.client = MultiServerMCPClient(connections=mcp_server_config)
        self.tools = None
        self.llm = load_model(backend_model=backend_model, temperature=temperature)
        self.memory_base_url = memory_base_url
        self.memory_workspace_id = memory_workspace_id
        self.enable_memory_tool = enable_memory_tool
        self.memory_retrieval_callback = memory_retrieval_callback

    def _build_retrieve_tool(self) -> StructuredTool:
        def retrieve_experience(query: str, reason: str, top_k: int = 3) -> dict:
            payload = {
                "workspace_id": self.memory_workspace_id,
                "query": query,
                "top_k": top_k,
            }
            try:
                response = requests.post(
                    url=f"{self.memory_base_url}retrieve_task_memory",
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                return {
                    "error": str(exc),
                    "query": query,
                    "reason": reason,
                }

            if self.memory_retrieval_callback is not None:
                try:
                    self.memory_retrieval_callback(result)
                except Exception as exc:
                    system_logger.warning(f"memory_retrieval_callback failed: {exc}")

            result["query"] = query
            result["reason"] = reason
            return result

        return StructuredTool.from_function(
            func=retrieve_experience,
            name="retrieve_experience",
            description=(
                "Retrieve relevant past troubleshooting experiences from ReMe when you are stuck "
                "or evidence is insufficient. Provide a concise query and a reason."
            ),
            args_schema=RetrieveExperienceInput,
        )

    async def load_tools(self):
        self.tools: list[StructuredTool] = await self.client.get_tools()
        for tool in self.tools:
            tool.handle_tool_error = True
            tool.handle_validation_error = True

        if self.enable_memory_tool:
            self.tools.append(self._build_retrieve_tool())

    def get_agent(self):
        agent = create_agent(
            model=self.llm,
            system_prompt=ON_DEMAND_DIAGNOSIS_PROMPT,
            tools=self.tools,
            name="DiagnosisAgentOnDemand",
        )
        return agent
