"""Specialized agent for extracting and updating agent self-cognition memories."""

from ..base_memory_agent import BaseMemoryAgent
from ...core.context import C
from ...core.enumeration import Role, MemoryType
from ...core.schema import Message
from ...core.utils import get_now_time, format_messages


@C.register_op()
class IdentitySummarizer(BaseMemoryAgent):
    """Analyzes conversations to extract and update agent's self-perception."""

    memory_type: MemoryType = MemoryType.IDENTITY

    async def build_messages(self) -> list[Message]:
        """Construct system and user messages with formatted context and timestamp."""
        system_prompt = self.prompt_format(
            prompt_name="system_prompt",
            now_time=get_now_time(),
            context=format_messages(self.get_messages()),
            memory_type=self.memory_type.value,
        )

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=self.get_prompt("user_message")),
        ]
        return messages

    async def _acting_step(self, assistant_message: Message, step: int, **kwargs) -> list[Message]:
        """Execute tool calls with workspace_id and author context."""
        return await super()._acting_step(assistant_message, step, author=self.author, **kwargs)
