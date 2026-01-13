"""Add history memory operation."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C
from ...core.enumeration import MemoryType
from ...core.schema import ToolCall, Message
from ...core.utils import format_messages


@C.register_op()
class AddHistoryMemory(BaseMemoryTool):
    """Add history memory from conversation messages."""

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "description": self.get_prompt("messages"),
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["messages"],
                },
            },
        )

    async def execute(self):
        messages: list[Message | dict] = self.context.get("messages", [])
        if not messages:
            self.output = "No messages provided for addition."
            return

        messages = [Message(**m) if isinstance(m, dict) else m for m in messages]
        memory_content = format_messages(messages)
        memory_node = self._build_memory_node(memory_content=memory_content, memory_type=MemoryType.HISTORY)

        vector_node = memory_node.to_vector_node()
        await self.vector_store.delete(vector_ids=[vector_node.vector_id])
        await self.vector_store.insert(nodes=[vector_node])

        self.output = "Successfully added history memory to vector_store."
        logger.info(self.output)
