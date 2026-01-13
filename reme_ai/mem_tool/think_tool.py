"""Think tool for agent reflection and planning.

This module provides a tool that prompts the model for explicit reflection
before taking actions, helping agents reason about their next steps.
"""

from .base_memory_tool import BaseMemoryTool
from ..core.context import C
from ..core.schema import ToolCall


@C.register_op()
class ThinkTool(BaseMemoryTool):
    """Utility that prompts the model for explicit reflection text.

    This tool provides a thinking mechanism for agents to reflect on:
    1. Whether current context is sufficient to answer
    2. What information is missing
    3. Which tool and parameters to use next
    """

    def __init__(self, add_output_reflection: bool = False, **kwargs):
        """Initialize the think tool.

        Args:
            add_output_reflection: If True, outputs the reflection content;
                                 if False, outputs a confirmation message
            **kwargs: Additional arguments passed to BaseOp
        """
        super().__init__(**kwargs)
        self.add_output_reflection: bool = add_output_reflection

    def _build_tool_call(self) -> ToolCall:
        """Build the tool call schema for think tool."""
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reflection": {
                            "type": "string",
                            "description": self.get_prompt("reflection"),
                        },
                    },
                    "required": ["reflection"],
                },
            },
        )

    async def execute(self):
        """Execute the think tool by processing reflection input."""
        if self.add_output_reflection:
            self.output = self.context["reflection"]
        else:
            self.output = self.get_prompt("reflection_output")
