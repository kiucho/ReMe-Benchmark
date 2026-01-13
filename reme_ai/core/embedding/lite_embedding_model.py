"""LiteLLM asynchronous embedding model implementation for ReMe.

Supports multiple embedding providers including Azure OpenAI, OpenAI, and others.
"""

import os

import litellm
from loguru import logger

from flowllm.core.context import C as FlowC

from .base_embedding_model import BaseEmbeddingModel
from ..context import C
from ..schema import VectorNode


@C.register_embedding_model("litellm")
@FlowC.register_embedding_model("litellm")
class LiteEmbeddingModel(BaseEmbeddingModel):
    """Async embedding model implementation using LiteLLM to support multiple providers."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        custom_llm_provider: str = "openai",
        api_version: str | None = None,
        **kwargs,
    ):
        """Initialize the LiteLLM embedding client with API configuration and provider settings.

        Args:
            api_key: API key for the embedding provider. Falls back to environment variables:
                     - For Azure: AZURE_EMBEDDING_API_KEY > AZURE_API_KEY > REME_EMBEDDING_API_KEY
                     - For others: REME_EMBEDDING_API_KEY
            base_url: Base URL for the API endpoint. Falls back to environment variables:
                      - For Azure: AZURE_EMBEDDING_API_BASE > AZURE_API_BASE > REME_EMBEDDING_BASE_URL
                      - For others: REME_EMBEDDING_BASE_URL
            custom_llm_provider: The embedding provider to use (e.g., "openai", "azure", "cohere").
            api_version: API version for Azure OpenAI (e.g., "2023-05-15").
                         Falls back to AZURE_EMBEDDING_API_VERSION > AZURE_API_VERSION.
            **kwargs: Additional arguments passed to BaseEmbeddingModel.
        """
        super().__init__(**kwargs)

        # For Azure, check Azure-specific environment variables first
        if custom_llm_provider == "azure":
            self.api_key = api_key or os.getenv("AZURE_EMBEDDING_API_KEY") or os.getenv("AZURE_API_KEY") or os.getenv("REME_EMBEDDING_API_KEY")
            self.base_url = base_url or os.getenv("AZURE_EMBEDDING_API_BASE") or os.getenv("AZURE_API_BASE") or os.getenv("REME_EMBEDDING_BASE_URL")
            self.api_version = api_version or os.getenv("AZURE_EMBEDDING_API_VERSION") or os.getenv("AZURE_API_VERSION")
        else:
            self.api_key = api_key or os.getenv("REME_EMBEDDING_API_KEY")
            self.base_url = base_url or os.getenv("REME_EMBEDDING_BASE_URL")
            self.api_version = api_version

        self.custom_llm_provider: str = custom_llm_provider

    def _build_embedding_kwargs(self, input_text: list[str], log_params: bool = True, **kwargs) -> dict:
        """Construct and log the parameters dictionary for LiteLLM embedding API calls."""
        embedding_kwargs = {
            "model": self.model_name,
            "input": input_text,
            "custom_llm_provider": self.custom_llm_provider,
            **self.kwargs,
            **kwargs,
        }

        # Add dimensions if specified
        if self.dimensions:
            embedding_kwargs["dimensions"] = self.dimensions

        # Add API key and base URL if provided
        if self.api_key:
            embedding_kwargs["api_key"] = self.api_key
        if self.base_url:
            embedding_kwargs["api_base"] = self.base_url

        # Add API version for Azure OpenAI
        if self.api_version:
            embedding_kwargs["api_version"] = self.api_version

        # Log parameters for debugging
        if log_params:
            log_kwargs: dict = {}
            for k, v in embedding_kwargs.items():
                if k == "input":
                    log_kwargs[k] = len(v) if v is not None else 0
                elif k == "api_key":
                    log_kwargs[k] = "***" if v else None
                else:
                    log_kwargs[k] = v
            logger.info(f"embedding_kwargs={log_kwargs}")

        return embedding_kwargs

    async def _get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float]]:
        """Fetch embeddings asynchronously from the API for a batch of strings."""
        embedding_kwargs = self._build_embedding_kwargs(input_text, **kwargs)
        response = await litellm.aembedding(**embedding_kwargs)

        # Extract embeddings from response, maintaining order by index
        # Handle both object attributes and dictionary keys (LiteLLM may return either)
        result_emb = [[] for _ in range(len(input_text))]
        for emb in response.data:
            if isinstance(emb, dict):
                # LiteLLM may return dictionaries in some cases
                idx = emb.get("index", 0)
                embedding = emb.get("embedding", [])
            else:
                # LiteLLM typically returns objects with attributes
                idx = emb.index
                embedding = emb.embedding
            result_emb[idx] = embedding
        return result_emb

    async def async_get_embeddings(self, query: str, **kwargs) -> list[float]:
        """Async method for FlowLLM's BaseVectorStore.async_get_embeddings compatibility.
        
        FlowLLM's BaseVectorStore calls this method for async embedding retrieval.
        
        Args:
            query: The text to get embeddings for.
            **kwargs: Additional arguments passed to the embedding API.
            
        Returns:
            A list of floats representing the embedding vector.
        """
        result = await self._get_embeddings([query], **kwargs)
        return result[0] if result else []

    async def async_get_node_embeddings(self, nodes: list[VectorNode], **kwargs) -> list[VectorNode]:
        """Async method for FlowLLM's BaseVectorStore.async_get_node_embeddings compatibility.
        
        FlowLLM's BaseVectorStore.async_insert calls this method for async node embedding.
        
        Args:
            nodes: List of VectorNode objects to get embeddings for.
            **kwargs: Additional arguments passed to the embedding API.
            
        Returns:
            List of VectorNode objects with populated vector fields.
        """
        contents = [node.content for node in nodes]
        embeddings = await self.get_embeddings(contents, **kwargs)
        
        if len(embeddings) == len(nodes):
            for node, vec in zip(nodes, embeddings):
                node.vector = vec
        else:
            logger.warning(f"Mismatch: got {len(embeddings)} vectors for {len(nodes)} nodes")
        return nodes
