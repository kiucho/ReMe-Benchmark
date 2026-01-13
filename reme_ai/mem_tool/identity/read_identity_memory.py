"""Read identity memory operation."""

from loguru import logger

from ..base_memory_tool import BaseMemoryTool
from ...core.context import C


@C.register_op()
class ReadIdentityMemory(BaseMemoryTool):
    """Read identity memory for agent self-cognition."""

    def __init__(self, **kwargs):
        kwargs["enable_multiple"] = False
        super().__init__(**kwargs)

    def _build_parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self):
        identity_memory = self.meta_memory.load("identity_memory") or ""
        self.output = identity_memory or "No identity memory found."
        logger.info(self.output)
