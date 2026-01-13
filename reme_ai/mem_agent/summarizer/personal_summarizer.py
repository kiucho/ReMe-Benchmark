"""Specialized agent for extracting and managing personal memories about specific individuals."""

from ..base_memory_agent import BaseMemoryAgent
from ...core.context import C
from ...core.enumeration import Role, MemoryType
from ...core.schema import Message
from ...core.utils import get_now_time, format_messages


@C.register_op()
class PersonalSummarizer(BaseMemoryAgent):
    """Extracts and stores personal information about individuals from conversations."""

    memory_type: MemoryType = MemoryType.PERSONAL

    async def build_messages(self) -> list[Message]:
        """Construct messages with context, memory_target, and memory_type information."""
        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=get_now_time(),
            context=self.description + "\n" + format_messages(self.get_messages()),
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
        )

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=self.get_prompt("user_message")),
        ]
        return messages

    async def _acting_step(self, assistant_message: Message, step: int, **kwargs) -> list[Message]:
        """Execute tool calls with memory_target, memory_type, and author context."""
        return await super()._acting_step(
            assistant_message,
            step,
            memory_target=self.memory_target,
            memory_type=self.memory_type.value,
            author=self.author,
            ref_memory_id=self.ref_memory_id,
            **kwargs,
        )
