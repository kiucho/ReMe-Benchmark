"""Orchestrator for complete memory summarization workflow across all memory types."""

import re
from typing import List

from loguru import logger

from ..base_memory_agent import BaseMemoryAgent
from ...core.context import C
from ...core.enumeration import Role
from ...core.schema import Message, MemoryNode
from ...core.utils import get_now_time, format_messages


@C.register_op()
class ReMeSummarizer(BaseMemoryAgent):
    """Coordinates memory updates by delegating to specialized memory agents."""

    def __init__(self, enable_tool_memory: bool = True, enable_identity_memory: bool = True, **kwargs):
        """Initialize with flags to enable/disable tool and identity memory processing."""
        super().__init__(**kwargs)
        self.enable_tool_memory = enable_tool_memory
        self.enable_identity_memory = enable_identity_memory

    async def _add_history_memory(self) -> MemoryNode:
        """Store conversation history and return the memory node."""
        from ...mem_tool import AddHistoryMemory

        op = AddHistoryMemory()
        await op.call(messages=self.get_messages())
        return op.output

    @staticmethod
    async def _read_identity_memory() -> str:
        """Retrieve agent's self-perception memory."""
        from ...mem_tool import ReadIdentityMemory

        op = ReadIdentityMemory()
        await op.call()
        return op.output

    async def _read_meta_memories(self) -> str:
        """Fetch all meta-memory entries that define specialized memory agents."""
        from ...mem_tool import ReadMetaMemory

        op = ReadMetaMemory(
            enable_tool_memory=self.enable_tool_memory,
            enable_identity_memory=self.enable_identity_memory,
        )
        await op.call()
        return str(op.output)

    async def build_messages(self) -> List[Message]:
        """Construct initial messages with context, identity, and meta-memory information."""
        memory_node: MemoryNode = await self._add_history_memory()
        self.context["ref_memory_id"] = memory_node.memory_id
        now_time = get_now_time()
        identity_memory = await self._read_identity_memory()
        meta_memory_info = await self._read_meta_memories()
        context = format_messages(self.get_messages())
        logger.info(
            f"now_time={now_time} "
            f"memory_node={memory_node} "
            f"identity_memory={identity_memory} "
            f"meta_memory_info={meta_memory_info} "
            f"context={context}",
        )

        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=now_time,
            identity_memory=identity_memory,
            meta_memory_info=meta_memory_info,
            context=context,
        )

        user_message = self.get_prompt("user_message")
        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_message),
        ]

        return messages

    async def _reasoning_step(self, messages: list[Message], step: int, **kwargs) -> tuple[Message, bool]:
        """Refresh meta-memory info in system prompt before each reasoning step."""
        meta_memory_info = await self._read_meta_memories()
        system_messages = [message for message in messages if message.role is Role.SYSTEM]
        if system_messages:
            system_message = system_messages[0]
            pattern = r'("- <memory_type>\(<memory_target>\): <description>"\n)(.*?)(\n\n)'
            replacement = rf"\g<1>{meta_memory_info}\g<3>"
            system_message.content = re.sub(pattern, replacement, system_message.content, flags=re.DOTALL)

        return await super()._reasoning_step(messages, step, **kwargs)

    async def _acting_step(self, assistant_message: Message, step: int, **kwargs) -> list[Message]:
        """Execute tool calls with ref_memory_id and author context."""
        return await super()._acting_step(
            assistant_message,
            step,
            ref_memory_id=self.context["ref_memory_id"],
            author=self.author,
            **kwargs,
        )
