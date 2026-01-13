"""Delete memory operation for vector store."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C


@C.register_op()
class DeleteMemory(BaseMemoryTool):
    """Delete memories from vector store by IDs.

    Supports single/multiple deletion modes via `enable_multiple` parameter.
    """

    def _build_parameters(self) -> dict:
        """Build input schema for single memory deletion."""
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
        """Build input schema for multiple memory deletion."""
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
        """Execute deletion: remove memories from vector store by IDs."""
        if self.enable_multiple:
            memory_ids = self.context.get("memory_ids", [])
        else:
            memory_id = self.context.get("memory_id", "")
            memory_ids = [memory_id] if memory_id else []

        # Filter out empty IDs
        memory_ids = [m for m in memory_ids if m]

        if not memory_ids:
            self.output = "No valid memory IDs provided for deletion."
            return

        await self.vector_store.delete(vector_ids=memory_ids)
        self.output = f"Successfully deleted {len(memory_ids)} memories from vector_store."
        logger.info(self.output)
