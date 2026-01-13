"""Asynchronous OpenAI-compatible LLM implementation supporting streaming, tool calls, and reasoning content."""

import os
from typing import AsyncGenerator

from loguru import logger
from openai import AsyncOpenAI

from .base_llm import BaseLLM
from ..context import C
from ..enumeration import ChunkEnum
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


@C.register_llm("openai")
class OpenAILLM(BaseLLM):
    """Asynchronous LLM client for OpenAI-compatible APIs supporting streaming completions and tool execution."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        """Initialize the OpenAI async client with API credentials and model configuration."""
        super().__init__(**kwargs)
        self.api_key: str = api_key or os.getenv("REME_LLM_API_KEY", "")
        self.base_url: str = base_url or os.getenv("REME_LLM_BASE_URL", "")

        # Create client using factory method
        self._client = self._create_client()

    def _create_client(self):
        """Create and return an instance of the AsyncOpenAI client."""
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _build_stream_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        log_params: bool = True,
        **kwargs,
    ) -> dict:
        """Construct the parameter dictionary for the OpenAI Chat Completions API call."""
        # Construct the API parameters by merging multiple sources
        llm_kwargs = {
            "model": self.model_name,
            "messages": [x.simple_dump() for x in messages],
            "tools": [x.simple_input_dump() for x in tools] if tools else None,
            "stream": True,
            **self.kwargs,
            **kwargs,
        }

        # Log parameters for debugging, with message/tool counts instead of full content
        if log_params:
            log_kwargs: dict = {}
            for k, v in llm_kwargs.items():
                if k in ["messages", "tools"]:
                    log_kwargs[k] = len(v) if v is not None else 0
                else:
                    log_kwargs[k] = v
            logger.info(f"llm_kwargs={log_kwargs}")

        return llm_kwargs

    async def _stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None,
        stream_kwargs: dict,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a stream of chat completion chunks including text, reasoning content, and tool calls."""
        stream_kwargs = stream_kwargs or {}
        completion = await self._client.chat.completions.create(**stream_kwargs)
        ret_tool_calls: list[ToolCall] = []

        async for chunk in completion:
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    yield StreamChunk(chunk_type=ChunkEnum.USAGE, chunk=chunk.usage.model_dump())
                    continue

            delta = chunk.choices[0].delta

            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                yield StreamChunk(chunk_type=ChunkEnum.THINK, chunk=delta.reasoning_content)

            if delta.content is not None:
                yield StreamChunk(chunk_type=ChunkEnum.ANSWER, chunk=delta.content)

            if delta.tool_calls is not None:
                for tool_call in delta.tool_calls:
                    self._accumulate_tool_call_chunk(tool_call, ret_tool_calls)

        for tool_data in self._validate_and_serialize_tools(ret_tool_calls, tools):
            yield StreamChunk(chunk_type=ChunkEnum.TOOL, chunk=tool_data)

    async def close(self):
        """Asynchronously close the OpenAI client and release network resources."""
        await self._client.close()
