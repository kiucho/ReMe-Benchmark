"""LiteLLM asynchronous implementation for ReMe."""

import os
from typing import AsyncGenerator

import litellm
from loguru import logger

from flowllm.core.context import C as FlowC

from .base_llm import BaseLLM
from ..context import C
from ..enumeration import ChunkEnum
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


@C.register_llm("litellm")
@FlowC.register_llm("litellm")
class LiteLLM(BaseLLM):
    """Async LLM implementation using LiteLLM to support multiple providers."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        custom_llm_provider: str = "openai",
        api_version: str | None = None,
        **kwargs,
    ):
        """Initialize the LiteLLM client with API configuration and provider settings.

        Args:
            api_key: API key for the LLM provider. Falls back to environment variables:
                     - For Azure: AZURE_LLM_API_KEY > AZURE_API_KEY > REME_LLM_API_KEY
                     - For others: REME_LLM_API_KEY
            base_url: Base URL for the API endpoint. Falls back to environment variables:
                      - For Azure: AZURE_LLM_API_BASE > AZURE_API_BASE > REME_LLM_BASE_URL
                      - For others: REME_LLM_BASE_URL
            custom_llm_provider: The LLM provider to use (e.g., "openai", "azure", "anthropic").
            api_version: API version for Azure OpenAI (e.g., "2024-02-01").
                         Falls back to AZURE_LLM_API_VERSION > AZURE_API_VERSION.
            **kwargs: Additional arguments passed to BaseLLM.
        """
        super().__init__(**kwargs)

        # For Azure, check Azure-specific environment variables first
        if custom_llm_provider == "azure":
            self.api_key = api_key or os.getenv("AZURE_LLM_API_KEY") or os.getenv("AZURE_API_KEY") or os.getenv("REME_LLM_API_KEY")
            self.base_url = base_url or os.getenv("AZURE_LLM_API_BASE") or os.getenv("AZURE_API_BASE") or os.getenv("REME_LLM_BASE_URL")
            self.api_version = api_version or os.getenv("AZURE_LLM_API_VERSION") or os.getenv("AZURE_API_VERSION")
        else:
            self.api_key = api_key or os.getenv("REME_LLM_API_KEY")
            self.base_url = base_url or os.getenv("REME_LLM_BASE_URL")
            self.api_version = api_version

        self.custom_llm_provider: str = custom_llm_provider

    def _build_stream_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        log_params: bool = True,
        **kwargs,
    ) -> dict:
        """Construct and log the parameters dictionary for LiteLLM API calls."""
        # Construct the API parameters by merging multiple sources
        llm_kwargs = {
            "model": self.model_name,
            "messages": [x.simple_dump() for x in messages],
            "tools": [x.simple_input_dump() for x in tools] if tools else None,
            "stream": True,
            "custom_llm_provider": self.custom_llm_provider,
            **self.kwargs,
            **kwargs,
        }

        # Add API key and base URL if provided
        if self.api_key:
            llm_kwargs["api_key"] = self.api_key
        if self.base_url:
            llm_kwargs["api_base"] = self.base_url

        # Add API version for Azure OpenAI
        if self.api_version:
            llm_kwargs["api_version"] = self.api_version

        # Log parameters for debugging, with message/tool counts instead of full content
        if log_params:
            log_kwargs: dict = {}
            for k, v in llm_kwargs.items():
                if k in ["messages", "tools"]:
                    log_kwargs[k] = len(v) if v is not None else 0
                elif k == "api_key":
                    # Mask API key in logs for security
                    log_kwargs[k] = "***" if v else None
                else:
                    log_kwargs[k] = v
            logger.info(f"llm_kwargs={log_kwargs}")

        return llm_kwargs

    async def _stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        stream_kwargs: dict | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Execute async streaming chat requests and yield processed response chunks."""
        stream_kwargs = stream_kwargs or {}
        completion = await litellm.acompletion(**stream_kwargs)
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

            if hasattr(delta, "tool_calls") and delta.tool_calls is not None:
                for tool_call in delta.tool_calls:
                    self._accumulate_tool_call_chunk(tool_call, ret_tool_calls)

        for tool_data in self._validate_and_serialize_tools(ret_tool_calls, tools):
            yield StreamChunk(chunk_type=ChunkEnum.TOOL, chunk=tool_data)

    async def achat(self, *args, **kwargs):
        """Alias for chat() to maintain compatibility with FlowLLM naming conventions.
        
        Some parts of the codebase expect 'achat' for async chat operations.
        This method simply delegates to the parent class's 'chat' method.
        """
        return await self.chat(*args, **kwargs)
