"""Hands-off tool for distributing memory tasks to appropriate agents."""

import json
from typing import TYPE_CHECKING

from loguru import logger

from .base_memory_tool import BaseMemoryTool
from ..core.context import C
from ..core.enumeration import MemoryType

if TYPE_CHECKING:
    from ..mem_agent import BaseMemoryAgent


@C.register_op()
class HandsOffTool(BaseMemoryTool):
    """Distribute memory tasks to appropriate agents based on memory_type."""

    def __init__(self, memory_agents: list["BaseMemoryAgent"], **kwargs):
        kwargs["sub_ops"] = memory_agents or []
        super().__init__(**kwargs)
        from ..mem_agent import BaseMemoryAgent

        self.sub_ops: list[BaseMemoryAgent] = [a for a in self.sub_ops if isinstance(a, BaseMemoryAgent)]

    @property
    def memory_agent_dict(self) -> dict[MemoryType, "BaseMemoryAgent"]:
        """Returns a dictionary mapping memory types to their corresponding agents."""
        return {a.memory_type: a for a in self.sub_ops}

    def _build_item_schema(self) -> tuple[dict, list[str]]:
        """Build shared schema properties and required fields for memory tasks."""
        properties = {
            "memory_type": {
                "type": "string",
                "description": self.get_prompt("memory_type"),
                "enum": [
                    MemoryType.IDENTITY.value,
                    MemoryType.PERSONAL.value,
                    MemoryType.PROCEDURAL.value,
                    MemoryType.TOOL.value,
                ],
            },
            "memory_target": {
                "type": "string",
                "description": self.get_prompt("memory_target"),
            },
        }
        required = ["memory_type", "memory_target"]
        return properties, required

    def _build_parameters(self) -> dict:
        """Build input schema for single memory task distribution."""
        properties, required = self._build_item_schema()
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_multiple_parameters(self) -> dict:
        """Build input schema for multiple memory task distribution."""
        item_properties, required_fields = self._build_item_schema()
        return {
            "type": "object",
            "properties": {
                "memory_tasks": {
                    "type": "array",
                    "description": self.get_prompt("memory_tasks"),
                    "items": {
                        "type": "object",
                        "properties": item_properties,
                        "required": required_fields,
                    },
                },
            },
            "required": ["memory_tasks"],
        }

    @staticmethod
    def _parse_memory_type_target(task: dict):
        memory_type = task.get("memory_type", "")
        memory_target = task.get("memory_target", "")
        return {
            "memory_type": MemoryType(memory_type),
            "memory_target": memory_target,
        }

    def _collect_tasks(self) -> list[dict]:
        """Collect memory tasks from context based on enable_multiple flag."""
        tasks: list[dict] = []
        if self.enable_multiple:
            memory_tasks: list[dict] = self.context.get("memory_tasks", [])
            for task in memory_tasks:
                tasks.append(self._parse_memory_type_target(task))
        else:
            tasks.append(self._parse_memory_type_target(self.context))
        return tasks

    async def execute(self):
        """Execute memory tasks by distributing to appropriate agents in parallel."""
        tasks = self._collect_tasks()

        if not tasks:
            self.output = "No valid memory tasks to execute."
            return

        # Submit tasks to corresponding agents
        agent_list = []
        for i, task in enumerate(tasks):
            memory_type: MemoryType = task["memory_type"]
            memory_target: str = task["memory_target"]

            if memory_type not in self.memory_agent_dict:
                logger.warning(f"No agent found for memory_type={memory_type}")
                continue

            agent_copy = self.memory_agent_dict[memory_type].copy()
            agent_list.append(
                {
                    "agent": agent_copy,
                    "memory_type": memory_type,
                    "memory_target": memory_target,
                },
            )

            logger.info(f"Task {i}: Submitting {memory_type.value} agent for target={memory_target}")
            self.submit_async_task(
                agent_copy.call,
                query=self.context.get("query", ""),
                messages=self.context.get("messages", []),
                memory_target=memory_target,
                ref_memory_id=self.context.get("ref_memory_id", ""),
            )

        await self.join_async_tasks()

        # Collect results
        results = []
        for i, (agent, memory_type, memory_target) in enumerate(agent_list):
            result_str = str(agent.output)
            results.append(
                {
                    "memory_type": memory_type.value,
                    "memory_target": memory_target,
                    "result": result_str[:200] + ("..." if len(result_str) > 200 else ""),
                },
            )
            logger.info(f"Task {i}: Completed {memory_type.value} agent for target={memory_target}")

        results_str = json.dumps(results, ensure_ascii=False, indent=2)
        self.output = f"Successfully executed {len(results)} memory tasks:\n{results_str}"
