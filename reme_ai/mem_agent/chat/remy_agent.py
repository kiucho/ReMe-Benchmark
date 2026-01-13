"""ReMy agent with identity and meta memory capabilities."""

from typing import List

from ..base_memory_agent import BaseMemoryAgent
from ...core.context import C
from ...core.enumeration import Role
from ...core.schema import Message
from ...core.utils import get_now_time


@C.register_op()
class ReMyAgent(BaseMemoryAgent):
    """Memory agent with identity awareness and meta memory retrieval."""

    def __init__(self, enable_tool_memory: bool = True, enable_identity_memory: bool = True, **kwargs):
        """Initialize ReMy agent with memory options."""
        super().__init__(**kwargs)
        self.enable_tool_memory = enable_tool_memory
        self.enable_identity_memory = enable_identity_memory

    @staticmethod
    async def _read_identity_memory() -> str:
        """Read and return identity memory as string."""
        from ...mem_tool import ReadIdentityMemory

        op = ReadIdentityMemory()
        await op.call()
        return str(op.output)

    async def _read_meta_memories(self) -> str:
        """Read and return meta memories as string."""
        from ...mem_tool import ReadMetaMemory

        op = ReadMetaMemory(
            enable_tool_memory=self.enable_tool_memory,
            enable_identity_memory=self.enable_identity_memory,
        )
        await op.call()
        return str(op.output)

    async def build_messages(self) -> List[Message]:
        """Build messages with system prompt and user messages."""
        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=get_now_time(),
            identity_memory=await self._read_identity_memory(),
            meta_memory_info=await self._read_meta_memories(),
        )

        return [Message(role=Role.SYSTEM, content=system_prompt)] + self.get_messages()
