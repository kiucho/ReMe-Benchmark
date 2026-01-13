"""Synchronous LiteLLM-based embedding model implementation for ReMe.

Supports multiple embedding providers including Azure OpenAI, OpenAI, and others.
This implementation provides sync-first methods for compatibility with FlowLLM's
MemoryVectorStore which sometimes calls embedding methods in sync contexts.
"""

import litellm
from flowllm.core.context import C as FlowC

from .lite_embedding_model import LiteEmbeddingModel
from ..context import C
from ..schema import VectorNode


@C.register_embedding_model("litellm_sync")
@FlowC.register_embedding_model("litellm_sync")
class LiteEmbeddingModelSync(LiteEmbeddingModel):
    """Synchronous embedding model implementation using LiteLLM to support multiple providers.
    
    This class overrides async methods to use sync implementations, ensuring compatibility
    with FlowLLM's MemoryVectorStore which may call methods in sync contexts without await.
    """

    def _get_embeddings_sync(self, input_text: list[str], **kwargs) -> list[list[float]]:
        """Fetch embeddings synchronously from the API for a batch of strings."""
        embedding_kwargs = self._build_embedding_kwargs(input_text, **kwargs)
        response = litellm.embedding(**embedding_kwargs)

        # Extract embeddings from response, maintaining order by index
        # Handle both object attributes and dictionary keys (LiteLLM may return either)
        result_emb = [[] for _ in range(len(input_text))]
        for emb in response.data:
            if isinstance(emb, dict):
                # LiteLLM sync returns dictionaries
                idx = emb.get("index", 0)
                embedding = emb.get("embedding", [])
            else:
                # LiteLLM async returns objects with attributes
                idx = emb.index
                embedding = emb.embedding
            result_emb[idx] = embedding
        return result_emb

    async def _get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float]]:
        """Override async method to use sync implementation for compatibility with FlowLLM."""
        return self._get_embeddings_sync(input_text, **kwargs)

    async def async_get_embeddings(self, query: str, **kwargs) -> list[float]:
        """Async method for FlowLLM's BaseVectorStore.async_get_embeddings compatibility.
        
        FlowLLM's BaseVectorStore calls this method for async embedding retrieval.
        """
        result = self._get_embeddings_sync([query], **kwargs)
        return result[0] if result else []

    def get_node_embeddings(self, nodes: list[VectorNode], **kwargs) -> list[VectorNode]:
        """Override as sync method for FlowLLM MemoryVectorStore.insert compatibility.
        
        FlowLLM's MemoryVectorStore.insert calls this method in a sync context without
        await, so this MUST be a regular method (not async) returning the result directly.
        """
        return self.get_node_embeddings_sync(nodes, **kwargs)

    async def async_get_node_embeddings(self, nodes: list[VectorNode], **kwargs) -> list[VectorNode]:
        """Async method for FlowLLM's BaseVectorStore.async_get_node_embeddings compatibility.
        
        FlowLLM's BaseVectorStore.async_insert calls this method for async node embedding.
        Uses sync implementation internally for consistency.
        """
        return self.get_node_embeddings_sync(nodes, **kwargs)
