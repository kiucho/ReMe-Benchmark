"""Read meta memory operation for retrieving memory metadata."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.enumeration import MemoryType


@C.register_op()
class ReadMetaMemory(BaseMemoryTool):
    """Read memory metadata (memory_type and memory_target) from meta storage.

    This operation reads stored memory metadata and optionally includes
    TOOL and IDENTITY type memories.
    """

    def __init__(
        self,
        enable_identity_memory: bool = False,
        **kwargs,
    ):
        """Initialize ReadMetaMemory.

        Args:
            enable_identity_memory: Include IDENTITY type meta memory. Defaults to False.
            **kwargs: Additional arguments for BaseMemoryTool.
        """
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)
        self.enable_identity_memory = enable_identity_memory

    def _build_parameters(self) -> dict:
        """Build input schema for reading meta memory.

        No input parameters required for reading.
        """
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def _load_meta_memories(self) -> list[dict[str, str]]:
        """Load meta memories from cache and apply filters."""
        result = self.meta_memory.load("meta_memories")
        all_memories = result if result is not None else []

        filtered_memories = []
        for m in all_memories:
            if m.get("memory_type") in [MemoryType.PERSONAL.value, MemoryType.PROCEDURAL.value]:
                filtered_memories.append(m)

        if self.enable_identity_memory:
            filtered_memories.append(
                {
                    "memory_type": MemoryType.IDENTITY.value,
                    "memory_target": "self",
                },
            )

        return filtered_memories

    def format_memory_metadata(self, memories: list[dict[str, str]]) -> str:
        """Format memory metadata into a readable string.

        Args:
            memories: List of memory metadata entries.

        Returns:
            str: Formatted memory metadata string.
        """
        if not memories:
            return ""

        lines = []
        for memory in memories:
            memory_type = memory["memory_type"]
            memory_target = memory["memory_target"]
            description = self.get_prompt(f"type_{memory_type}")
            lines.append(f"- {memory_type}({memory_target}): {description}")

        return "\n".join(lines)

    async def execute(self):
        """Execute the read meta memory operation.

        Reads memory metadata from cache storage and formats output.
        """
        memories = self._load_meta_memories()

        if memories:
            self.output = self.format_memory_metadata(memories)
            logger.info(f"Retrieved {len(memories)} meta memory entries")
        else:
            self.output = "No memory metadata found."
            logger.info(self.output)
