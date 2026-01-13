"""Read history memory operation."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.schema import MemoryNode


@C.register_op()
class ReadHistoryMemory(BaseMemoryTool):
    """Read history memories by IDs."""

    def _build_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": self.get_prompt("memory_id"),
                },
            },
            "required": ["memory_id"],
        }

    def _build_multiple_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "memory_ids": {
                    "type": "array",
                    "description": self.get_prompt("memory_ids"),
                    "items": {"type": "string"},
                },
            },
            "required": ["memory_ids"],
        }

    async def execute(self):
        if self.enable_multiple:
            memory_ids: list[str] = self.context.get("memory_ids", [])
        else:
            memory_id = self.context.get("memory_id", "")
            memory_ids: list[str] = [memory_id] if memory_id else []

        memory_ids = [mid for mid in memory_ids if mid]

        if not memory_ids:
            self.output = "No valid history memory IDs provided for reading."
            logger.warning(self.output)
            return

        nodes = await self.vector_store.get(vector_ids=memory_ids)

        if not nodes:
            self.output = "No history memories found with the provided IDs."
            logger.warning(self.output)
            return

        memories: list[MemoryNode] = [MemoryNode.from_vector_node(n) for n in nodes]
        self.output = "---\n".join([m.content for m in memories])
        logger.info(f"Successfully read {len(memories)} history memories.")
