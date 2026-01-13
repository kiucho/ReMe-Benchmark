"""Update identity memory operation."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C


@C.register_op()
class UpdateIdentityMemory(BaseMemoryTool):
    """Update identity memory for agent self-cognition."""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def _build_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "identity_memory": {
                    "type": "string",
                    "description": self.get_prompt("identity_memory"),
                },
            },
            "required": ["identity_memory"],
        }

    async def execute(self):
        identity_memory = self.context.get("identity_memory", "")

        if not identity_memory:
            self.output = "No valid identity memory provided for update."
            logger.warning(self.output)
            return

        self.meta_memory.save("identity_memory", identity_memory)
        self.output = "Successfully updated identity memory."
        logger.info(self.output)
