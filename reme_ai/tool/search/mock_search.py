"""Mock search tool for testing purposes.

This module provides a mock search operation that generates simulated
search results using an LLM, useful for testing without making actual API calls.
"""

import json
import random

from loguru import logger

from ...core.context import C
from ...core.enumeration import Role
from ...core.op import BaseOp
from ...core.schema import ToolCall, Message
from ...core.utils import extract_content


@C.register_op()
class MockSearch(BaseOp):
    """Operation for generating mock search results.

    This operation generates simulated search results using an LLM,
    useful for testing and development without requiring actual search API access.
    """

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "description": self.get_prompt("tool"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "query",
                        },
                    },
                    "required": ["query"],
                },
            },
        )

    async def execute(self):
        query: str = self.context.query
        num_results: int = random.randint(0, 5)
        messages = [
            Message(
                role=Role.SYSTEM,
                content="You are a helpful assistant that generates realistic search results in JSON format.",
            ),
            Message(
                role=Role.USER,
                content=self.prompt_format("mock_search_prompt", query=query, num_results=num_results),
            ),
        ]

        logger.info(f"messages={messages}")

        def callback_fn(message: Message):
            return extract_content(message.content, "json")

        search_results: str = await self.llm.chat(messages=messages, callback_fn=callback_fn)
        self.output = json.dumps(search_results, ensure_ascii=False, indent=2)
