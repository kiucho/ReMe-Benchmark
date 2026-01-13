"""ReMe retriever that builds messages with meta memories."""

from typing import List

from ..base_memory_agent import BaseMemoryAgent
from ...core.context import C
from ...core.enumeration import Role
from ...core.schema import Message
from ...core.utils import get_now_time, format_messages


@C.register_op()
class ReMeRetriever(BaseMemoryAgent):
    """Memory agent that retrieves and builds messages with meta memory context."""

    def __init__(self, meta_memories: list[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.meta_memories: list[dict] = meta_memories

    @staticmethod
    async def _read_meta_memories() -> str:
        """Read and return meta memories as string."""
        from ...mem_tool import ReadMetaMemory

        op = ReadMetaMemory(enable_identity_memory=False)
        await op.call()
        return str(op.output)

    async def build_messages(self) -> List[Message]:
        """Build messages with system prompt and user message."""
        from ...mem_tool import ReadMetaMemory

        if self.meta_memories:
            meta_memory_info = ReadMetaMemory().format_memory_metadata(self.meta_memories)
        else:
            meta_memory_info = await self._read_meta_memories()

        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=get_now_time(),
            meta_memory_info=meta_memory_info,
            context=format_messages(self.get_messages()),
        )

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=self.get_prompt("user_message")),
        ]
        return messages
